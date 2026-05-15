# PS-GPPO 租卡训练工作流

## 0. 当前代码完整性结论

当前代码有两条线：

- `trainer/train_psgppo.py`：面向 MiniMind RLAIF 数据的 sequence-level PS-GPPO 训练入口。当前数据没有真实 multi-turn `turn_ids`，所以这里是 “one completion = one step” 的版本。
- `Agentic RL/PS-GPPO`：面向真正 Agentic RL 的 package 原型，包含 `turn_ids`、multi-turn rollout、GAE、calculator smoke test、pytest 单测。

可以上服务器跑的主线是第一条：`train_psgppo.py`。它已经具备训练闭环：数据、full-SFT 权重、old/ref policy、reward model scoring、K inner updates、W&B logging、PS-GPPO loss helper、smoke test 入口都齐了。

还没完全闭合的部分：

- 本机没有 `torch`，所以本地只完成了 `py_compile` 静态检查，没跑数值 smoke。
- 本地没有找到 `internlm2-1_8b-reward` reward model 目录；服务器上需要先下载到本地路径，再传给 `--reward_model_path`。
- sequence-level PS-GPPO 可以作为 MiniMind/RLAIF 实验，不要把它表述成完整 multi-turn Agentic RL 结果。真正 Agentic 版本需要跑 `Agentic RL/PS-GPPO` 的 calculator/GSM8K 或自定义 multi-turn env。

## 1. 租卡选择

### 推荐优先级

1. 省心国内平台：AutoDL / Featurize / 恒源云。
2. 价格敏感或需要全球节点：RunPod / Vast.ai。
3. 只做 2-3 个 smoke steps：RTX 4090 24GB 足够。
4. 要跑完整 3k steps、`num_generations=6`、`max_gen_len=1024`：优先 L40S 48GB / RTX 6000 Ada 48GB / A100 40GB 或 80GB。

### 最保守配置

- GPU：L40S 48GB 或 A100 80GB。
- CPU：8 核以上。
- 内存：64GB 以上。
- 系统盘/数据盘：至少 100GB；如果要缓存 reward model、W&B、多个 checkpoint，建议 150GB。
- 镜像：PyTorch 2.1+ / CUDA 12.1+ / Python 3.10+。如果用 50 系显卡，选 PyTorch 2.7/2.8 + CUDA 12.8。

### 低成本 smoke 配置

- GPU：RTX 4090 24GB。
- 启动命令里把 `--batch_size 1 --num_generations 2 --max_gen_len 64 --max_train_steps 2`。

## 2. 服务器模板

### AutoDL

推荐选平台内置 PyTorch 镜像，优先：

- PyTorch 2.1.0 / Python 3.10 / CUDA 12.1
- 或 PyTorch 2.5.1 / Python 3.12 / CUDA 12.4
- 50 系显卡才考虑 PyTorch 2.7/2.8 / CUDA 12.8

不要用 conda 安装 CPU-only torch。优先用平台自带 torch；不满足时用 pip 安装 CUDA wheel。

### RunPod

选 PyTorch 官方/平台 PyTorch template，GPU 选 L40S / RTX 6000 Ada / A100。把代码和模型放到 persistent volume，避免关机丢数据。

### Vast.ai

选带 PyTorch + CUDA 的 Docker template。创建实例前把 disk slider 调到 100GB+，Vast 创建后磁盘大小不方便改。模板里 Docker image 要有明确 tag，不要空 tag。

## 3. 上传代码与模型

服务器目录建议：

```bash
/workspace/gppo/
  Clipping-Aware-Policy-Optimization-Comparison-Suite/
  models/
  outputs/
```

从本机上传：

```bash
rsync -av --progress \
  ./Clipping-Aware-Policy-Optimization-Comparison-Suite/ \
  user@server:/workspace/gppo/Clipping-Aware-Policy-Optimization-Comparison-Suite/
```

如果平台有网盘/数据盘，也可以先打包再上传：

```bash
cd /path/to/local/projects
tar --exclude='**/wandb' --exclude='**/__pycache__' --exclude='**/.git' \
  -czf adaptive_clipping_minimind.tar.gz "Clipping-Aware-Policy-Optimization-Comparison-Suite"
```

## 4. 环境配置

进入 MiniMind 项目：

```bash
cd /workspace/gppo/Clipping-Aware-Policy-Optimization-Comparison-Suite
python -V
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

安装依赖：

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

如果没有 torch 或 torch 不能用 CUDA，再按平台 CUDA 版本装 CUDA wheel，例如 CUDA 12.1：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果平台已经有可用 torch，不要重复安装 torch。

## 5. 下载 reward model

`LMForRewardModel` 当前强制 offline local path，所以必须先下载到本地目录。

ModelScope：

```bash
mkdir -p /workspace/gppo/models
python - <<'PY'
from modelscope import snapshot_download
path = snapshot_download(
    "Shanghai_AI_Laboratory/internlm2-1_8b-reward",
    cache_dir="/workspace/gppo/models"
)
print(path)
PY
```

把打印出来的路径记为：

```bash
export REWARD_MODEL_PATH="/workspace/gppo/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward"
```

如果实际路径不同，以 `snapshot_download` 打印的路径为准。

HuggingFace 备用：

```bash
huggingface-cli download internlm/internlm2-1_8b-reward \
  --local-dir /workspace/gppo/models/internlm2-1_8b-reward
export REWARD_MODEL_PATH="/workspace/gppo/models/internlm2-1_8b-reward"
```

## 6. 先跑 objective smoke

```bash
cd /workspace/gppo/Clipping-Aware-Policy-Optimization-Comparison-Suite
python trainer/smoke_test_psgppo_sequence.py
```

期望看到：

```text
PS-GPPO sequence objective smoke test OK
loss=...
step_ratio_mean=...
```

## 7. 再跑 2-step 训练 sanity check

```bash
cd /workspace/gppo/Clipping-Aware-Policy-Optimization-Comparison-Suite
python trainer/train_psgppo.py \
  --from_weight full_sft \
  --save_weight psgppo_smoke \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --batch_size 1 \
  --num_generations 2 \
  --num_workers 0 \
  --max_gen_len 64 \
  --max_train_steps 2 \
  --save_interval 100000 \
  --wandb_mode disabled
```

如果这一步过了，再开 W&B。

## 8. 正式训练命令

固定 epsilon 版本：

```bash
cd /workspace/gppo/Clipping-Aware-Policy-Optimization-Comparison-Suite
wandb login
python trainer/train_psgppo.py \
  --from_weight full_sft \
  --save_weight psgppo_fixed_eps005 \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --use_wandb \
  --wandb_project ClipComparison \
  --wandb_mode online \
  --batch_size 2 \
  --num_generations 6 \
  --num_policy_updates 4 \
  --learning_rate 3e-7 \
  --beta 0.1 \
  --epsilon_s 0.05 \
  --beta1 0.75 \
  --beta2 1.0 \
  --max_seq_len 768 \
  --max_gen_len 1024 \
  --grad_clip 1.0 \
  --save_interval 50
```

长度自适应 epsilon 版本：

```bash
python trainer/train_psgppo.py \
  --from_weight full_sft \
  --save_weight psgppo_len_scaled \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --use_wandb \
  --wandb_project ClipComparison \
  --wandb_mode online \
  --batch_size 2 \
  --num_generations 6 \
  --num_policy_updates 4 \
  --learning_rate 3e-7 \
  --beta 0.1 \
  --epsilon_s_mode length_scaled \
  --token_epsilon 0.2 \
  --epsilon_s_min 0.01 \
  --epsilon_s_max 0.10 \
  --max_seq_len 768 \
  --max_gen_len 1024 \
  --grad_clip 1.0 \
  --save_interval 50
```

建议先跑 `fixed_eps005` 和 `len_scaled` 两个版本，不要一上来扩太多 ablation。

## 9. 监控指标

W&B 重点看：

- `stability/reward_ma`：主曲线，是否从负值上升并稳定。
- `train/reward`：原始 reward，有噪声正常。
- `train/kl_ref` / `stability/kl_ma`：漂移控制。
- `train/entropy_mean` / `stability/entropy_ma`：是否过早塌缩。
- `clip/ratio_std`：token ratio 扩散程度。
- `clip/step_ratio_mean/std/p10/p90`：PS-GPPO 关键指标。
- `clip/step_in_clip_frac`：太高说明约束没起作用，太低说明 clip 太紧。
- `clip/step_silenced_frac`：过高说明很多 out-of-clip wrong-sign 被静默，需调 epsilon 或 lr。
- `clip/w_mean` / `clip/w_nonzero_frac`：权重是否爆炸或大面积归零。
- `train/grad_norm`：是否经常撞上 grad clip。

经验判断：

- reward_ma 上不去，KL/ratio 很稳：学习太保守，增大 epsilon 或 lr。
- reward 上升但 entropy 快速下降：减小 lr、提高 KL beta，或尝试 `entropy_coef=0.005/0.01`。
- step_silenced_frac > 15%：epsilon 太窄或 lr 太大。
- step_in_clip_frac 几乎 100%：epsilon 太宽，PS-GPPO 退化成普通 sequence surrogate。

## 10. 导出与分析曲线

训练后从 W&B 导出 CSV，至少包含：

- reward_ma
- reward
- kl_ma / kl_ref
- entropy_ma / entropy_mean
- ratio_std
- step_ratio_std
- step_in_clip_frac
- step_silenced_frac
- w_mean
- grad_norm
- avg_response_len

最小分析表：

```python
import pandas as pd

df = pd.read_csv("wandb_export.csv")
tail = df.tail(200)
summary = tail.mean(numeric_only=True).sort_index()
print(summary)
```

建议最终比较：

1. `psgppo_fixed_eps005` vs `psgppo_len_scaled`
2. 两者再和已有 `GRPO / GSPO / CE-GPPO` 对齐比较
3. 只总结 final-window 后 200 或 500 steps，避免被早期冷启动噪声误导

## 11. 出问题时的排查顺序

1. `torch.cuda.is_available()` 是不是 True。
2. `full_sft_768.pth` 是否在 `out/`。
3. `REWARD_MODEL_PATH` 是否是本地目录，里面是否有 `config.json` 和 safetensors。
4. 先降到 `batch_size=1, num_generations=2, max_gen_len=64`。
5. 若 reward model OOM，先把 `max_gen_len` 降到 128/256。
6. 若 W&B 网络失败，加 `--wandb_mode offline`，训练后再 `wandb sync`。
7. 若生成太慢，用更短 `max_gen_len` 或租 48GB/80GB 卡。
