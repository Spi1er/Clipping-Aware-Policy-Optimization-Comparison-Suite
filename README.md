# Clipping-Aware Policy Optimization Comparison Suite

A research-oriented comparison suite for studying **clipping-aware policy optimization** in **LLM post-training**, with a focus on **PPO/GRPO-style objectives**, **adaptive clipping strategies**, and **training dynamics under a shared experimental setup**.

This repository is built to compare how different clipping designs influence optimization behavior in reasoning-oriented reinforcement learning for language models, including effects on stability, gradient flow, reward trends, and clipped-token behavior.

---

## Motivation

In modern LLM post-training, policy optimization methods such as PPO and GRPO rely heavily on clipped objectives to stabilize updates. However, clipping is not just a numerical trick — it directly shapes:

- which tokens continue contributing gradients,
- how aggressively the policy is allowed to move,
- whether optimization becomes too conservative or too unstable,
- and how reward improvement trades off against training robustness.

This project studies those effects in a controlled implementation framework by comparing multiple clipping-related variants under matched training settings.

---

## Project Goals

This repository aims to:

- build a unified experimental framework for **PPO/GRPO-style clipping comparisons**;
- implement and compare multiple clipping-aware training variants;
- analyze how clipping affects **token-level update behavior**, **optimization stability**, and **training dynamics**;
- provide a practical codebase for future extensions in **reasoning-oriented RL post-training**.

---

## What This Repository Focuses On

Instead of presenting a full-scale production post-training pipeline, this repository focuses on the **algorithmic and engineering layer** of policy optimization, especially:

- clipping-based objective design,
- adaptive clipping mechanisms,
- gradient behavior under clipped updates,
- training stability under shared hyperparameter settings,
- and reproducible comparison across multiple variants.

---

## Implemented / Compared Methods

The current codebase is organized around a shared trainer framework and includes multiple training variants for comparison, including:

- **PPO-style clipped policy optimization**
- **GRPO-style training variants**
- **GSPO-style / generalized clipping variants**
- **DAPO-style or adaptive clipping inspired variants**
- additional experimental variants for clipping-aware objective comparison

> Note: The repository emphasizes a **unified comparison framework**. Exact implementations may evolve as experiments are refined and aligned more closely with their source papers.

---

## Experimental Framing

This project is designed around a shared comparison setup:

- same base code structure,
- aligned training scripts across variants,
- matched or comparable hyperparameter settings where possible,
- shared logging / analysis workflow,
- and emphasis on observing **training curves and optimization behavior**, not only final scores.

Key analysis targets include:

- reward trend,
- loss behavior,
- gradient norm behavior,
- clipping ratio / clipped-token behavior,
- stability across training steps,
- and practical engineering issues during RL post-training.

---

## Repository Structure

```text
.
├── trainer/                 # main training scripts and trainer utilities
├── model/                   # model implementation and related modules
├── scripts/                 # helper scripts / inference / evaluation tools
├── minimind-3/              # lightweight model-related configs/assets (without large weights)
├── requirements.txt         # dependencies
└── README.md
