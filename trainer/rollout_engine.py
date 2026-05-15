"""Rollout Engine - 可插拔的推理引擎
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


# ===== 计算每个 token 的 logprob =====
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


# ===== Rollout 结果 =====
@dataclass
class RolloutResult:
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]


# ===== Rollout 引擎抽象基类 =====
class RolloutEngine(ABC):
    tokenizer = None
    
    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        pass
    
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass


# ===== PyTorch 原生推理引擎 =====
class TorchRolloutEngine(RolloutEngine):
    """
    Rollout engine that keeps a FROZEN SNAPSHOT of the policy (π_old) separate
    from the live trainable model.

    Design:
      • _old_policy  — deep-copied, eval(), requires_grad_(False).
                       Used for BOTH generation and old-logp computation.
                       Updated by calling update_policy() AFTER each outer
                       optimisation step.
      • The live trainable model is never touched here; the engine only reads
        its state_dict during update_policy().

    Why this matters:
      Within one outer step the trainable model is frozen at π_N.  After K
      gradient updates it becomes π_{N+K}.  The next rollout uses _old_policy
      which was synced to π_N at the end of the previous outer step.  So the
      new-logps (from π_{N+K}) and old-logps (from π_N) are genuinely different,
      giving ratio = π_{N+K}/π_N ≠ 1 and allowing PPO clipping to activate.
    """

    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.tokenizer    = tokenizer
        self.device       = device
        self.autocast_ctx = autocast_ctx
        # Create frozen snapshot; the live model is NOT stored here.
        self._old_policy  = self._make_snapshot(policy_model)

    @staticmethod
    def _make_snapshot(model: torch.nn.Module) -> torch.nn.Module:
        """
        Deep-copy model into a standalone eval/no-grad snapshot.
        Unwraps DDP and torch.compile wrappers so the copy is a plain nn.Module.
        """
        import copy
        raw  = model.module if isinstance(model, DistributedDataParallel) else model
        raw  = getattr(raw, '_orig_mod', raw)   # unwrap torch.compile
        snap = copy.deepcopy(raw)
        snap.eval()
        snap.requires_grad_(False)
        return snap

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # Always use the FROZEN snapshot — never the live trainable model.
        old_model = self._old_policy

        with torch.no_grad():
            output_ids = old_model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_generations,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )  # [B*num_gen, P+R]

        # generate() runs under torch.inference_mode() internally, so output_ids is an
        # inference tensor.  Inference tensors cannot be saved by autograd (raised as
        # "Inference tensors cannot be saved for backward"), which crashes the training
        # forward pass when output_ids is used as an index in gather() on a grad-tracked
        # logits tensor.  Clone once here to produce a normal tensor; completion_ids is
        # derived from this clone so it is clean as well.
        output_ids = output_ids.clone()

        prompt_len     = prompt_ids.size(1)
        completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]

        from contextlib import nullcontext
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        # Compute old-logps from the FROZEN snapshot under no_grad.
        # These are the reference log-probs π_old(a|s) used for the IS ratio.
        with torch.no_grad(), ctx:
            per_token_logps = compute_per_token_logps(old_model, output_ids, completion_ids.size(1))

        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions)

    def update_policy(self, model: torch.nn.Module):
        """
        Sync the frozen old-policy snapshot with the current trainable model.

        Call this AFTER each outer optimisation step (after all K inner gradient
        updates).  Uses load_state_dict for an efficient in-place weight copy
        rather than a second deepcopy.  eval() and requires_grad_(False) are
        preserved across load_state_dict.
        """
        raw = model.module if isinstance(model, DistributedDataParallel) else model
        raw = getattr(raw, '_orig_mod', raw)
        self._old_policy.load_state_dict(raw.state_dict())


# ===== SGLang HTTP API 推理引擎 =====
class SGLangRolloutEngine(RolloutEngine):
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # 去除左侧 padding tokens，只保留有效 token
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
        
        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,
        }
        
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        
        results = resp.json()
        if not isinstance(results, list):
            results = [results]
        
        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        prompt_len = prompt_ids.size(1)
        
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)
            
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
        
        device = prompt_ids.device
        max_out_len = max(len(ids) for ids in all_output_ids)
        max_comp_len = max(len(ids) for ids in all_completion_ids)
        max_logp_len = max(len(lp) for lp in all_logprobs)
        
        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)
        
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len),
            per_token_logps=pad_to_tensor(all_logprobs, max_logp_len, pad_val=0.0),
            completions=completions,
        )
    
    def update_policy(self, model: torch.nn.Module):
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        abs_path = os.path.abspath(self.shared_ckpt_path)
        unwrapped.lm_head.weight = torch.nn.Parameter(unwrapped.lm_head.weight.clone())
        state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
        unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
        unwrapped.model.embed_tokens.weight = unwrapped.lm_head.weight
        self.tokenizer.save_pretrained(abs_path)
        resp = self.http.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": abs_path},
            timeout=self.timeout
        )
        if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
        return resp.status_code == 200
    
    def flush_cache(self) -> bool:
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200
    
    def health(self) -> bool:
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


# ===== 工厂函数 =====
def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
