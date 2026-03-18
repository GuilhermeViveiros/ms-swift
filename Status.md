# TowerVision GKD — Project Status

## Research Goal

On-policy knowledge distillation to improve **multimodal synergy** in TowerVision models:
- Vision skills should increase
- Text/multilingual capabilities must be preserved or improved
- **Key signal:** text-only scores (ALM-Bench) stay flat or improve **while** vision scores (OCRBench, TextVQA, etc.) go up

**Longer-term hypothesis:** Self-distillation via EMA teacher — improve a frontier VLM without external supervision, using cross-modal consistency as a free self-supervised reward.

---

## Model Setup

| Role    | Model           | State at training start |
|---------|-----------------|-------------------------|
| Student | TowerVision-2B  | Post-PE only: MLP projector trained, LLM backbone frozen, **no instruction tuning** |
| Teacher | TowerVision-9B  | Post-PE + full SFT (strong vision + multilingual) |

---

## Training Framework

- **Trainer:** `swift/rlhf_trainers/gkd_trainer.py`
- **Loss:** JSD with on-policy mixing (`lmbda`)
- **Key hyperparameters:**
  - `beta = 0.5` — JSD interpolation coefficient
  - `lmbda = 0.5` — on-policy mixing probability
  - `lmbda_warmup_ratio = 0.3` — ramp lmbda 0 → 0.5 over first 30% of training
- **Infrastructure:** 28-node SLURM, 4 GPUs/node, vLLM colocated, DeepSpeed ZeRO-2
- **Script:** `examples/train/multimodal/rlhf/gkd/full_sbatch.sh`

---

## Training Runs History

| Version | Checkpoints | Outcome |
|---------|-------------|---------|
| v0-20260316-020701 | ckpt-5494 | Early checkpoints, seemingly OK |
| v1-20260316-112802 | ckpt-8241 → ckpt-10988+ | **Collapsed at step ~9155–9160** (see below) |
| v4-20260318-013350 | ckpt-1806+ | Healthy: loss ~0.09–0.14, grad_norm ~0.4–1.4, lmbda warmup working |
| job-288248 (2026-03-18) | — | **Test run (1k samples/dataset)** — validated all fixes: loss 0.22→0.14, grad_norm 1–2, lmbda warmup correct, completion_len non-zero throughout ✅ |
| job-288872 (2026-03-18) | — | **Full run** — lmbda=0.5, warmup=0.3, full dataset. In progress. |
| (2026-03-18) | — | **Ablation: lmbda=0** — pure off-policy baseline. In progress. |

---

## Collapse Diagnosis (v1 run)

**What was observed:**
- Early checkpoints (ckpt-5494, ckpt-8241) were generating reasonable outputs
- Later checkpoints (post ckpt-10988) generated **nothing** at inference
- Training logs showed `loss = 0.0` and a perfectly constant `grad_norm = 2.44948983` (= sqrt(6)) — frozen optimizer artifact

**Root cause chain:**
```
lmbda=0.5 → 50% of steps use student on-policy generations
    ↓
Student has no proficiency in the target distribution → incoherent/empty on-policy completions
(generalizes beyond instruction tuning: any domain the base model hasn't seen —
new language, new format, new modality — will trigger the same collapse)
    ↓
Empty completions → all labels = -100 → num_valid = 0 → loss returns 0.0 silently
    ↓
Zero loss → zero gradient → model parameters freeze
    ↓
Frozen model keeps generating empty completions → loop locked at step ~9155
    ↓
Run resumed from already-collapsed ckpt-10988 → 7000+ more dead steps
```

**Code-level cause:** `generalized_jsd_loss` in `gkd_trainer.py:498` returns `zeros(())` when `num_valid == 0` — silent, no warning.

**Full analysis:** `docs/gkd_training_collapse_analysis.md`

---

## Fixes Applied (current run)

1. **`lmbda_warmup_ratio = 0.3`** — start with `lmbda=0` (pure off-policy SFT distillation), linearly ramp to `lmbda=0.5` over first 30% of training. Student first builds enough competence in the target distribution before contributing on-policy signal. Applies broadly: any scenario where the student has low proficiency in the target domain (new language, new modality, no instruction tuning, etc.).

2. **Completion length tracking** — `completion_len/mean`, `min`, `max` logged to wandb every step. This metric would have caught the collapse at step ~9155 before loss dropped.

3. **`ZeroLossEarlyStoppingCallback`** — stops training if `loss <= 1e-8` for `patience` consecutive logging steps (default: 5). Controlled via `--zero_loss_patience N`.

4. **`max_completion_length` 324 → 524** — reduces premature EOS token learning.

5. **Resumed from ckpt-8241** (last known good), not the collapsed ckpt-10988.

---

## Known Open Issues

- **Multi-turn label masking bug:** `_prepare_batch_inputs` is called the same way for both on-policy and off-policy samples, labeling all assistant turns in both cases. The correct behavior depends on sample type:
  - **Off-policy** (dataset target): label all assistant turns — full multi-turn supervision is correct
  - **On-policy** (student-generated): label only the last turn — student only produced that response; earlier assistant turns should be masked (`-100`)
  Fix requires branching `_prepare_batch_inputs` (or post-processing labels) based on whether the sample is on-policy or off-policy. Also: remove stray `pdb.set_trace()` in `training_step`.
-**Gemma 2 + flash_attn2 NaN risk:** Gemma 2's softcap attention is not natively supported by standard flash_attn2 kernels, which can produce NaN logits. Fix: use `attn_implementation="kernels-community/flash-attn2"` (community kernel with proper softcap support). Currently using standard `flash_attn2`; should migrate to the community kernel.

---

## Current Status

- **Full run submitted** (job-288872) — lmbda=0.5 + warmup=0.3, full dataset, all fixes applied
- **lmbda=0 ablation submitted** — pure off-policy baseline, same dataset/hyperparams
- **Eval:** Pending — waiting for checkpoints from current runs

---

## Datasets

Vision-language mixture:
- `llava-next-cc-ocr-multi-lan` (multilingual OCR)
- `dvqa`, `plotqa`, `tabmwp` (chart/table understanding)
- `gemini-{aokvqa,chartqa,docvqa,iconqa,infographic-vqa,rlaif-4v,textvqa}` (filtered)
- `okvqa`, `st_vqa`, `tally_qa`, `r1-vision-ai2d`
- `VisionBlocks-pixmo-{ask-model-anything,cap-qa}`
- `euroblocks-sft-0525-text-only` (text-only, for language preservation)
