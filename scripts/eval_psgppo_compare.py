import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="Repository root. Defaults to the parent of this script directory.")
    parser.add_argument("--data_path", default="dataset/rlaif.jsonl")
    parser.add_argument("--reward_model_path", required=True)
    parser.add_argument("--weights", nargs="+", default=["full_sft", "psgppo_run1"])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--thinking_ratio", type=float, default=0.9)
    parser.add_argument("--out_dir", default="/workspace/gppo/eval_psgppo_run1")
    args = parser.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from dataset.lm_dataset import RLAIFDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.shared_rl_utils import calculate_rewards, rep_penalty
    from trainer.trainer_utils import LMForRewardModel, setup_seed

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(root / "model")
    tokenizer.padding_side = "left"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds = RLAIFDataset(
        str(root / args.data_path),
        tokenizer,
        max_length=args.max_seq_len,
        thinking_ratio=args.thinking_ratio,
    )
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), min(args.num_samples, len(ds)))

    prompts = []
    for idx in indices:
        # Seed per index so both compared weights see exactly the same prompt text.
        random.seed(args.seed + idx)
        prompts.append({"index": idx, "prompt": ds[idx]["prompt"]})

    reward_model = LMForRewardModel(args.reward_model_path, device=device, dtype=torch.float16)

    def load_model(weight_name):
        cfg = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len + args.max_new_tokens,
        )
        model = MiniMindForCausalLM(cfg)
        ckpt = root / "out" / f"{weight_name}_{args.hidden_size}.pth"
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        # Keep MiniMind in fp32 for this evaluation. The custom generate()
        # implementation can occasionally produce NaN probabilities in fp16.
        return model.eval().to(device)

    def generate_for_weight(weight_name):
        setup_seed(args.seed)
        model = load_model(weight_name)
        records = []
        start = time.time()
        for offset in range(0, len(prompts), args.batch_size):
            batch = prompts[offset : offset + args.batch_size]
            batch_prompts = [x["prompt"] for x in batch]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["input_ids"] = inputs["input_ids"][:, -args.max_seq_len :]
            inputs["attention_mask"] = inputs["attention_mask"][:, -args.max_seq_len :]
            prompt_len = inputs["input_ids"].size(1)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                )
            completion_ids = gen[:, prompt_len:]
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            rewards = calculate_rewards(batch_prompts, completions, 1, device, reward_model).detach().cpu().tolist()

            for item, comp_ids, comp, reward in zip(batch, completion_ids.detach().cpu(), completions, rewards):
                ids = comp_ids.tolist()
                eos = tokenizer.eos_token_id in ids
                trimmed_ids = ids[: ids.index(tokenizer.eos_token_id) + 1] if eos else ids
                answer = comp.split("</think>", 1)[-1].strip() if "</think>" in comp else comp.strip()
                records.append(
                    {
                        "weight": weight_name,
                        "index": item["index"],
                        "reward": float(reward),
                        "token_len": len(trimmed_ids),
                        "char_len": len(comp.strip()),
                        "hit_max_len": (not eos and len(ids) >= args.max_new_tokens),
                        "has_think_end": "</think>" in comp,
                        "think_end_count": comp.count("</think>"),
                        "rep_penalty": float(rep_penalty(answer)),
                        "prompt": item["prompt"],
                        "completion": comp,
                    }
                )
            done = min(offset + args.batch_size, len(prompts))
            print(f"[{weight_name}] {done}/{len(prompts)} done", flush=True)
        elapsed = time.time() - start
        del model
        torch.cuda.empty_cache()
        return records, elapsed

    all_records = []
    elapsed = {}
    for weight in args.weights:
        recs, seconds = generate_for_weight(weight)
        elapsed[weight] = seconds
        all_records.extend(recs)

    records_path = out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_weight = {w: [r for r in all_records if r["weight"] == w] for w in args.weights}
    summary = {"num_samples": len(prompts), "elapsed_seconds": elapsed, "weights": {}}
    for w, recs in by_weight.items():
        def mean(key):
            return sum(r[key] for r in recs) / max(len(recs), 1)
        rewards = [r["reward"] for r in recs]
        summary["weights"][w] = {
            "reward_mean": mean("reward"),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "token_len_mean": mean("token_len"),
            "char_len_mean": mean("char_len"),
            "hit_max_len_frac": mean("hit_max_len"),
            "has_think_end_frac": mean("has_think_end"),
            "exactly_one_think_end_frac": sum(r["think_end_count"] == 1 for r in recs) / max(len(recs), 1),
            "rep_penalty_mean": mean("rep_penalty"),
        }

    if len(args.weights) == 2:
        a, b = args.weights
        pairs = {}
        for r in by_weight[a]:
            pairs.setdefault(r["index"], {})[a] = r
        for r in by_weight[b]:
            pairs.setdefault(r["index"], {})[b] = r
        diffs = [v[b]["reward"] - v[a]["reward"] for v in pairs.values() if a in v and b in v]
        summary["pairwise"] = {
            "delta_reward_mean": sum(diffs) / max(len(diffs), 1),
            "win_rate_second_over_first": sum(d > 0 for d in diffs) / max(len(diffs), 1),
            "tie_rate": sum(d == 0 for d in diffs) / max(len(diffs), 1),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_path = out_dir / "sample_pairs.md"
    with sample_path.open("w", encoding="utf-8") as f:
        f.write("# Evaluation Sample Pairs\n\n")
        for idx in indices[:12]:
            pair = {r["weight"]: r for r in all_records if r["index"] == idx}
            f.write(f"## index {idx}\n\n")
            for w in args.weights:
                r = pair[w]
                f.write(f"### {w} reward={r['reward']:.4f} len={r['token_len']} hit_max={r['hit_max_len']}\n\n")
                f.write(r["completion"].strip()[:1200] + "\n\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"records={records_path}", flush=True)
    print(f"samples={sample_path}", flush=True)


if __name__ == "__main__":
    main()
