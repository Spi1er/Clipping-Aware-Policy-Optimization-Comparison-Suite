# Per-Step Gradient-Preserving Policy Optimization (PS-GPPO)

A clipping method designed for agentic RL, derived from the empirical findings of the
ClipComparison suite (GRPO / DAPO / DCPO / CFPO / SAPO / CE-GPPO / GSPO on MiniMind).

---

## 1. What Problem Are We Solving?

Standard clipping methods (GRPO, DAPO, etc.) were designed for **single-turn RLHF**:
one prompt → one completion → one reward signal.
Agentic RL adds structure that breaks two implicit assumptions:

### Problem A — The ratio product explosion

In single-turn RLHF, the token-level IS weight for one completion is:

```
r_{1:L} = π_θ(y_1) · π_θ(y_2) · … · π_θ(y_L)
           ────────────────────────────────────
           π_old(y_1) · π_old(y_2) · … · π_old(y_L)
```

The clipping objective works on each `r_t = π_θ(y_t) / π_old(y_t)` individually.

In our experiments the per-token ratio had p10 ≈ 0.95, p90 ≈ 1.05 (σ ≈ 0.04 at
steady state, ≈ 0.12 early in training).  That looks stable.

Now consider a 10-step agent trajectory where each step has 30 action tokens:

```
Early training (σ_token ≈ 0.12):
  Var[log r_t]   = 0.0144   (single token)
  Var[log w_τ]   = 300 · 0.0144 = 4.32   → σ_traj ≈ 2.08
  95% range of w_τ ≈ [e^{-4.2}, e^{4.2}] = [0.015, 66]

Late training (σ_token ≈ 0.04):
  Var[log w_τ]   = 300 · 0.0016 = 0.48   → σ_traj ≈ 0.69
  95% range of w_τ ≈ [0.25, 3.9]
```

Token-level clipping at ε = 0.2 does **not** prevent the trajectory-level IS weight
from exploding — it only clips each token's contribution independently.  The product
still varies over orders of magnitude, causing high-variance gradient estimates that
destabilize training.  This is what killed GSPO (which tried to fix it at
full-trajectory level) and severely damaged SAPO early in training (ratio std ≈ 32
vs. ≈ 0.12 for GRPO).

### Problem B — Dead zones waste sparse gradient signal

GRPO's min-clip objective sets gradient = 0 for tokens where `r_t ∉ [1-ε, 1+ε]`:

```
∂L_GRPO / ∂θ = 0   when r_t > 1+ε  or  r_t < 1-ε
```

In single-turn RLHF this wastes about 1–2% of tokens per step (clipped fraction from
experiments).  In agentic RL with sparse terminal rewards, the gradient signal is
already thin.  Silencing even 1% of tokens that happen to fall outside the clip window
removes information the policy urgently needs during the cold-start phase.

From experiment quartile data, all methods spent Q1 (first 25% of training) at reward
≈ −1.2 to −1.4.  CE-GPPO — the only method that kept gradient signal alive for
out-of-clip tokens — escaped negative reward fastest and achieved the highest reward.

---

## 2. The Two Building Blocks

PS-GPPO combines two existing ideas at a new granularity.

### Building block 1: Geometric-mean step ratio (from GSPO, applied per-step)

GSPO (2507.18071) proposed normalising the IS weight over the full trajectory.
It failed because the exponent `1/L_total` is too small for large models: small ε
on a sequence of 300+ tokens means the policy can barely update.

The fix is to apply the geometric mean **per agent step**, not per full trajectory:

```
s_k(θ)  =  exp( 1/L_k · Σ_{t=1}^{L_k} log[ π_θ(y_{k,t} | ctx_{k,t}) / π_old(y_{k,t} | ctx_{k,t}) ] )

         =  ( Π_{t=1}^{L_k} r_{k,t} )^{1/L_k}
```

`s_k` is the **geometric mean of token-level ratios within one step**.

By the CLT, if individual log-ratios are iid with variance σ²:

```
Var[log s_k] = σ² / L_k
```

The variance shrinks inversely with step length.  Longer agent actions are *more*
stable, not less, because the geometric mean averages out token-level noise.

Comparison of ratio distributions for a 30-token step:

```
                     Token level (r_t)    Step level (s_k, L_k=30)
σ (early training)      0.12                  0.12/√30 ≈ 0.022
95% range              [0.76, 1.28]            [0.957, 1.045]
σ (late training)       0.04                  0.04/√30 ≈ 0.0073
95% range              [0.92, 1.08]            [0.986, 1.014]
```

The step-level ratio lives **within a well-behaved range** at all training stages.
ε_s can be calibrated to match a target clipping fraction without the scale mismatch
that broke GSPO.

### Building block 2: Gradient-preserving clip (from CE-GPPO)

CE-GPPO (2509.20712) identifies two regions that GRPO silences but shouldn't:

- **NA&LP** (Negative Advantage, Low Probability): `r_t < 1−ε` AND `Â < 0`
  The policy moved *away* from a bad action.  That's good — GRPO zeroes the gradient
  here, meaning it doesn't reinforce that the policy is moving in the right direction.

- **PA&LP** (Positive Advantage, Large Probability): `r_t > 1+ε` AND `Â > 0`
  The policy moved *toward* a good action even more aggressively.  GRPO also zeroes
  this, blocking the exploration signal.

CE-GPPO reintroduces these regions with bounded gradient magnitudes:

```
In-clip:  l_t =  r_t · Â                                (standard surrogate)
NA&LP:    l_t =  β₁ · (1−ε) / sg(r_t) · r_t · Â        (bounded: grad ∝ β₁·(1−ε)·Â)
PA&LP:    l_t =  β₂ · (1+ε) / sg(r_t) · r_t · Â        (bounded: grad ∝ β₂·(1+ε)·Â)
```

where `sg(·)` is stop-gradient.  Setting β₁ < 1 slows over-exploitation of unlikely
bad actions; β₂ ≥ 1 preserves the exploration gradient for confidently good actions.
In the experiments, CE-GPPO with β₁=0.75, β₂=1.0 was the best-performing method.

---

## 3. PS-GPPO: Full Mathematical Definition

### Setup

```
Agent trajectory:  τ = (s₀, a₀, r₀, s₁, a₁, r₁, …, s_K, a_K, r_K)

Step k action:     aₖ = (y_{k,1}, …, y_{k,L_k})   L_k tokens

Context at token t of step k:
  ctx_{k,t} = [s₀, a₀, …, sₖ, y_{k,1}, …, y_{k,t−1}]

Step-level advantage:   Â_k  (from GAE, MC returns, or process reward model)
```

### Step-level ratio

```
s_k(θ) = exp( 1/L_k · Σ_{t=1}^{L_k} log π_θ(y_{k,t} | ctx_{k,t}) − log π_old(y_{k,t} | ctx_{k,t}) )
```

Equivalently, using token-level log-probabilities already computed in the forward pass:

```
s_k(θ) = exp( [Σ_t (logp_θ_{k,t} − logp_old_{k,t}) · mask_{k,t}] / [Σ_t mask_{k,t}] )
```

where `mask_{k,t}` is 1 for valid tokens in step k (before EOS), 0 otherwise.

### Region classification

```
ε_s  — step-level clip radius  (see §4 for calibration)

Rₖᶦⁿ  = { k  :  s_k ∈ [1 − ε_s,  1 + ε_s] }                          (in-clip)
Rₖᴺᴬ  = { k  :  s_k < 1 − ε_s   AND  Â_k < 0 }                       (NA&LP)
Rₖᴾᴬ  = { k  :  s_k > 1 + ε_s   AND  Â_k > 0 }                       (PA&LP)
Rₖˢⁱˡ = { k  :  remaining steps }    (out-of-clip, wrong-sign advantage — silenced)
```

### Per-step objective

```
                  ┌  s_k · Â_k                               k ∈ Rₖᶦⁿ
                  │
L_k(θ)  =        │  β₁ · (1−ε_s) / sg(s_k) · s_k · Â_k    k ∈ Rₖᴺᴬ
                  │
                  │  β₂ · (1+ε_s) / sg(s_k) · s_k · Â_k    k ∈ Rₖᴾᴬ
                  │
                  └  0                                         k ∈ Rₖˢⁱˡ
```

### KL regularisation (applied at token level within each step)

```
KL_k = Σ_t mask_{k,t} · [ exp(logp_ref_{k,t} − logp_θ_{k,t}) − (logp_ref_{k,t} − logp_θ_{k,t}) − 1 ]
       ──────────────────────────────────────────────────────────────────────────────────────────────────
                                      Σ_t mask_{k,t}
```

### Full training objective

```
                   1   K
J_PS-GPPO(θ)  =  ─── · Σ  [ L_k(θ)  −  β · KL_k ]
                   K  k=1
```

Gradient with respect to θ for step k at token t:

```
∂L_k / ∂θ  =

  Â_k · ∇_θ log π_θ(y_{k,t} | ctx_{k,t})               k ∈ Rₖᶦⁿ  (unscaled)

  β₁ · (1−ε_s) · Â_k · ∇_θ log π_θ(y_{k,t} | ctx_{k,t})    k ∈ Rₖᴺᴬ  (attenuated)

  β₂ · (1+ε_s) · Â_k · ∇_θ log π_θ(y_{k,t} | ctx_{k,t})    k ∈ Rₖᴾᴬ  (scaled)

  0                                                        k ∈ Rₖˢⁱˡ
```

The gradient is zero only for the `Rˢⁱˡ` region (out-of-clip, wrong-sign advantage).
This is the correct region to silence: the policy has moved in the wrong direction
relative to the advantage signal.

### Special cases / backward compatibility

```
β₁ = β₂ = 0,  L_k = 1   →   GRPO (standard PPO token-level min-clip)
β₁ = β₂ = 1,  L_k = 1   →   CE-GPPO at token level
β₁ = β₂ = 0,  L_k = L   →   GSPO (full-trajectory geometric mean)
ε_s → ∞,      any L_k   →   unclipped REINFORCE at step level
```

---

## 4. Calibrating ε_s

`ε_s` is **not** the same as GRPO's token-level `ε_t`.  They live on different scales.

### Theoretical relationship

If individual log-ratios are approximately Gaussian with std σ_token:

```
log s_k  ~  N(μ, σ_token² / L_k)

P(s_k > 1 + ε_s)  ≈  P(Z > log(1+ε_s) · √L_k / σ_token)
```

To match GRPO's ~2% per-token clipping rate at the step level:

```
ε_s  ≈  ε_token / √L_k   (rough first-order approximation)

Example:  ε_token = 0.2,  L_k = 30  →  ε_s ≈ 0.037
          ε_token = 0.2,  L_k = 10  →  ε_s ≈ 0.063
```

### Practical calibration procedure

1. Run the first N steps of training logging `s_k` (easy — it's just mean of log-ratios).
2. Measure the empirical std of `log s_k` across steps.
3. Set `ε_s = 2.0 × observed_std` to target ~5% step-level clipping rate.
4. Treat it as a tunable hyperparameter in [ε_token/√L_max, ε_token/√L_min].

### Why ε_s should be small

From experiments: token-level ratio std dropped from ~0.12 early to ~0.06 late.
For a 30-token step this gives step-level ratio std ≈ 0.011–0.022.
A step-level ε_s = 0.05 would clip almost nothing (which is fine — PS-GPPO's main
benefit is gradient preservation, not clip frequency).

---

## 5. Intuition: What Each Region Means for an Agent

Consider an agent that has been writing Python to solve a task.

```
Old policy (π_old):  "calls search() with keyword='temperature'"  → reward −0.5
New policy (π_θ):    "calls search() with keyword='climate'"      → reward +0.8
Advantage Â_k = +1.3 (this step improved on the old approach)
```

**Case A: s_k = 1.08, ε_s = 0.05  →  PA&LP region (Â > 0, s_k > 1+ε_s)**

The policy has shifted substantially toward the better argument ("climate").
GRPO: gradient = 0.  The policy doesn't know it should keep moving this direction.
PS-GPPO: gradient = β₂ · 1.05 · Â_k · ∇log π  (preserved, bounded by β₂·(1+ε_s))

This is the key benefit for exploration in agentic RL: the policy learns that "going
further toward that search term" is correct even when it's already outside the window.

**Case B: s_k = 0.91, ε_s = 0.05  →  NA&LP region (Â < 0, s_k < 1−ε_s)**

The policy is moving away from a bad tool call.  This is CORRECT behavior.
GRPO: gradient = 0.  The policy gets no signal that moving away was right.
PS-GPPO: gradient = β₁ · 0.95 · Â_k · ∇log π  (preserved but attenuated)
β₁ < 1 limits how hard it reinforces "moving away" — important when the old policy
was correct and the advantage estimate is noisy (common with sparse terminal rewards).

**Case C: s_k = 1.08, Â_k < 0  →  Silenced region**

The policy moved toward an action that turned out to have negative advantage.
Both GRPO and PS-GPPO: gradient = 0.  Correct — don't reinforce a bad shift.

**The step-level ratio adds a fourth dimension of intuition:**
s_k = 0.92 doesn't mean "every token in this step was 8% less likely."
It means "on average across this step's tokens, the policy has moved away from π_old."
Individual tokens may have s > 1 or s < 1, but the net effect of the whole action
has drifted.  This is the semantically meaningful unit for an agent's behavior.

---

## 6. Reasonable Cases to Use PS-GPPO

### Strong fit

| Scenario | Why PS-GPPO fits |
|----------|-----------------|
| **Multi-step tool-use agent** (5–50 steps, terminal or process reward) | Core use case. Step-level ratio prevents product explosion. Gradient preservation helps during cold-start. |
| **Code generation agent** (write, test, debug loop) | Each step (write function / run tests / fix error) is a natural unit. Advantages from test pass/fail signal. |
| **Reasoning chain with verifiable steps** (math, logic) | Each reasoning step is a step k. PRM scores give per-step Â_k. |
| **Early-stage agentic training** (cold start, mostly failing) | Gradient preservation keeps learning signal alive when most trajectories fail. |
| **Long action tokens per step** (L_k ≥ 20) | Geometric mean stabilises most here — the normalization gives the biggest variance reduction. |

### Partial fit (consider alternatives)

| Scenario | Concern | Alternative |
|----------|---------|-------------|
| **Very short steps** (L_k ≤ 5 tokens per action) | Step-level ≈ token-level; less benefit from geometric mean | CE-GPPO at token level is likely sufficient |
| **Single-turn RLHF** (no multi-step structure) | PS-GPPO reduces to CE-GPPO with one step | Use CE-GPPO directly |
| **Extremely sparse reward** (1 bit per episode) | Advantage estimate quality is poor; β₁/β₂ gradient scaling may amplify noise | Pair with GAE (γ=0.99, λ=0.95) or process RM |
| **Very large models** (30B+) | Full-trajectory GSPO was designed for this regime | GSPO with tuned ε_s; but test PS-GPPO first |

### Do not use

| Scenario | Reason |
|----------|--------|
| **Sequence-level language model** with no step boundaries | Step structure is undefined; s_k reduces to full-sequence GSPO which failed here |
| **Dense token-level process reward** (reward per token) | Token-level CE-GPPO is the right granularity; step-level aggregation loses precision |

---

## 7. Hyperparameter Guide

| Parameter | Role | Suggested start | Tuning direction |
|-----------|------|----------------|-----------------|
| `ε_s` | Step-level clip radius | `ε_token / √L_k_avg` | ↑ if step clipping fraction > 10%; ↓ if < 0.5% |
| `β₁` | NA&LP gradient scale | 0.75 (from CE-GPPO best result) | ↓ if policy is unstable / KL diverges; ↑ if learning stalls |
| `β₂` | PA&LP gradient scale | 1.0 | ↑ (e.g. 1.1) if entropy collapses early; ↓ if gradient norms spike |
| `β` (KL coef) | KL penalty | 0.1 (same as baseline) | Same as GRPO — this is the KL from ref, unrelated to clip |
| `K` (inner updates) | Number of gradient steps per rollout | 4 | Same rationale as GRPO: K ≥ 2 needed for s_k ≠ 1 |

---

## 8. Implementation Sketch

The change relative to CE-GPPO is minimal: replace token-level ratios with per-step
geometric means before the region classification.

```python
# In the K-epoch inner loop, after forward pass:
# per_token_logps:  [num_steps, L_max]
# old_per_token_logps: [num_steps, L_max]  (from rollout engine, fixed for all K)
# step_mask:        [num_steps, L_max]     1 for valid tokens, 0 for padding/post-EOS
# step_advantages:  [num_steps]            one advantage per step

log_ratio     = per_token_logps - old_per_token_logps          # [num_steps, L_max]
step_len      = step_mask.sum(dim=-1).clamp(min=1).float()     # [num_steps]

# Geometric-mean step ratio
log_s         = (log_ratio * step_mask).sum(dim=-1) / step_len  # [num_steps]
s_k           = torch.exp(log_s)                                # [num_steps]

adv           = step_advantages                                 # [num_steps]

# Region classification
in_clip = (s_k >= 1 - eps_s) & (s_k <= 1 + eps_s)
na_lp   = (s_k <  1 - eps_s) & (adv < 0)
pa_lp   = (s_k >  1 + eps_s) & (adv > 0)

# Step-level loss weight
w = torch.ones_like(s_k)
w[na_lp] = beta1 * (1 - eps_s) / s_k[na_lp].detach()
w[pa_lp] = beta2 * (1 + eps_s) / s_k[pa_lp].detach()
# Silenced region (out-of-clip, wrong-sign adv): w stays 1 but Â·w = 0 naturally
# because these steps have adv * w = adv (gradient = 0 only if we zero w explicitly)
# Correct implementation: explicitly zero the silenced region
silenced     = (~in_clip) & (~na_lp) & (~pa_lp)
w[silenced]  = 0.0

# Broadcast step weight back to tokens
w_token = w.unsqueeze(-1).expand_as(step_mask)                 # [num_steps, L_max]

# Token-level KL (unchanged from GRPO)
per_token_kl, kl_val = compute_kl_terms(ref_logps, per_token_logps, step_mask)

# Token-level loss with step-level weight
per_token_loss = -(s_k.unsqueeze(-1) * adv.unsqueeze(-1) - beta * per_token_kl)
                                                                # [num_steps, L_max]
per_token_loss = per_token_loss * w_token

# Aggregate: mean over tokens in step, then mean over steps
loss = ((per_token_loss * step_mask).sum(dim=-1)
        / step_len).mean()
```

### What to log (add to W&B schema)

```python
# Step-level ratio diagnostics
"clip/step_ratio_mean"   : s_k.mean().item()
"clip/step_ratio_std"    : s_k.std().item()
"clip/step_ratio_p10"    : s_k.quantile(0.10).item()
"clip/step_ratio_p90"    : s_k.quantile(0.90).item()
"clip/step_in_clip_frac" : in_clip.float().mean().item()
"clip/step_na_lp_frac"   : na_lp.float().mean().item()
"clip/step_pa_lp_frac"   : pa_lp.float().mean().item()
"clip/step_silenced_frac": silenced.float().mean().item()
```

These mirror the `clip/` schema from the existing suite so they are directly
comparable on the same W&B dashboard.

---

## 9. Connection to Existing Methods (Relationship Map)

```
                          IS weight granularity
                    token-level          step-level       traj-level
                  ┌────────────────┬─────────────────┬──────────────────┐
  Hard clip       │ GRPO / DAPO    │                 │   GSPO (failed)  │
                  │ DCPO (adaptive)│                 │                  │
  ─────────────── ├────────────────┼─────────────────┼──────────────────┤
  Gradient-       │ CE-GPPO        │  PS-GPPO  ◀─────│                  │
  preserving      │ (token-level)  │  (proposed)     │                  │
  ─────────────── ├────────────────┼─────────────────┼──────────────────┤
  No clip         │ CFPO           │                 │                  │
                  │ (quadratic)    │                 │                  │
  ─────────────── ├────────────────┼─────────────────┼──────────────────┤
  Soft gate       │ SAPO           │                 │                  │
                  │ (sigmoid)      │                 │                  │
                  └────────────────┴─────────────────┴──────────────────┘

PS-GPPO occupies the empty cell that combines:
  • GSPO's scale-normalisation idea (geometric mean) at the right granularity (step)
  • CE-GPPO's gradient-preservation insight (no dead zones) at that same granularity
```

---

## 10. What to Expect from the Experiments

Based on the ClipComparison suite findings:

```
Predicted ranking for multi-step agentic RL (5–20 steps, sparse terminal reward):

1. PS-GPPO              — gradient-preserving + stable ratio distribution per step
2. CE-GPPO (token)      — gradient-preserving but vulnerable to ratio product growth
3. DAPO / CFPO          — solid baselines, but token-level dead zones or no step norm
4. GRPO                 — baseline; dead zones and ratio accumulation both hurt
5. DCPO                 — over-permissive bounds destabilise training (observed)
6. SAPO                 — soft gate decouples exploration from learning (observed)
7. GSPO (full traj)     — scale mismatch; unlikely to recover without careful ε tuning

Key diagnostic signals to watch:

• clip/step_ratio_std early in training: should be ~0.02–0.04 (not 30+ like SAPO/GSPO)
• clip/step_silenced_frac: target 1–5% (too high = information loss; 0% = no constraint)
• stability/entropy_ma: should stay ≥ 1.5 nats; if it drops below 1.0, reduce β₂
• stability/reward_ma Q1 → Q2 transition: PS-GPPO should exit negative reward faster
  than CE-GPPO due to more stable early-training ratio distribution
```
