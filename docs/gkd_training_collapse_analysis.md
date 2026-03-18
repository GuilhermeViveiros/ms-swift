# GKD Training Collapse Analysis
**Run:** `distill_tvision_28_beta_0.5_lmbda_0.5_TowerVision-9B` (v1-20260316-112802)
**Date:** 2026-03-17

---

## Project Context

**Goal:** Understand whether on-policy distillation can improve multimodal synergy — preserving or
increasing the student's text capabilities while increasing its vision skills.

**Model setup:**
- **Student:** TowerVision-2B after the **PE (Projector Encoding) stage only**. During PE, only the
  MLP projector was trained; the LLM backbone and vision encoder were completely frozen.
  The student LLM has **never seen instruction-following data** — it has zero chat/instruction
  capability at the start of GKD training.
- **Teacher:** TowerVision-9B after **PE + full SFT** with the entire model fine-tuned — fully
  instruction-tuned, strong in both vision and multilingual text.
- **Hypothesis:** On-policy distillation from the 9B SFT teacher should build instruction-following
  and vision skills simultaneously, without the text degradation typical of standard vision SFT.

**Why JSD loss starts low (~0.1) despite the large capability gap:**
The JSD is computed over completion token distributions. The student's pretrained LLM backbone already
assigns reasonable probability to natural language tokens (including the teacher's responses), since
both models share the same tokenizer and base vocabulary statistics. The instruction-following
*quality* differs enormously, but the raw token-level distribution overlap is high enough to produce
low JSD at the start. This is expected.

**Critical implication for on-policy mixing (`lmbda`):**
Since the student has no instruction-following capability at the start, its on-policy generations
are completely incoherent — not just weak on vision, but random text. With `lmbda=0.5`, 50% of
training steps use these incoherent completions as training targets, immediately risking the
feedback loop that caused the collapse. **`lmbda` must be kept very low early in training.**

---

## Summary

The final checkpoints (post ~ckpt-10988) produce empty outputs at inference time.
The root cause is a **JSD loss collapse to 0.0** triggered by an on-policy feedback loop,
after which the model parameters froze and all subsequent training was ineffective.

---

## Evidence

### 1. Loss = 0.0 throughout the entire last job (job 283557)

From `logs/gkd_tvision_full/gkd_tvision_full_283557.out`:
```
{'loss': 0.0, 'grad_norm': 2.44948983, 'learning_rate': 3.77e-06, 'global_step': '10990/18311'}
...
{'loss': 0.0, 'grad_norm': 2.44948983, 'learning_rate': 0.0,      'global_step': '18311/18311'}
```

- Loss was **exactly 0.0 from the first logged step** (10990, right after resuming from ckpt-10988)
- `grad_norm` was a **perfectly constant `2.44948983` (= sqrt(6))** throughout — a frozen optimizer artifact
- `train_loss: 0.0` in the final summary confirms the entire resumed run was a no-op

### 2. Exact collapse step confirmed: **9155 → 9160** (job 281128)

From `logs/gkd_tvision_full/gkd_tvision_full_281128.out`:
```
# Last non-zero loss:
{'loss': 0.10582101, 'grad_norm': 2.44948983, 'epoch': 0.5, 'global_step': '9155/18311'}

# First zero loss (next logging window, 5 steps later):
{'loss': 0.0,        'grad_norm': 2.44948983, 'epoch': 0.5, 'global_step': '9160/18311'}
```

- Collapse happened around **step 9155–9160**, ~epoch 0.5
- From that point the model trained with `loss=0.0` through steps 9160 → 18311
- The run was then resumed from ckpt-10988 (already collapsed), adding 7000+ more dead steps
- **ckpt-8241** (epoch ~0.45) is confirmed as the last checkpoint before the collapse window

### 3. Eval scores reflect the degradation

ALM-Bench accuracy (ckpt-5494 → ckpt-8241), mixed signals already visible:

| Language | ckpt-5494 | ckpt-8241 | Delta |
|----------|-----------|-----------|-------|
| pt       | 30.6%     | 12.9%     | -17.7 |
| ko       | 43.1%     | 29.2%     | -13.9 |
| zht      | 28.8%     | 17.3%     | -11.5 |
| hg       | 21.9%     | 14.1%     | -7.8  |
| hi       | 19.2%     | 11.5%     | -7.7  |
| it       | 18.3%     | 28.3%     | +10.0 |
| uk       | 2.9%      | 11.4%     | +8.5  |

OCRBench: 29.8% → 30.6% (marginal improvement, still OK at ckpt-8241).

The large drops in some languages at ckpt-8241 suggest the collapse was already beginning
before that checkpoint.

### 4. The code-level cause: `num_valid == 0` in `generalized_jsd_loss`

In `swift/rlhf_trainers/gkd_trainer.py`, line 498:
```python
if num_valid == 0:
    return student_logits.new_zeros(())  # silent zero loss
```

When the student generates empty/EOS-only completions, all labels become `-100`.
After the label shift in `compute_loss`:
```python
shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
mask = shifted_labels != -100
```
`mask` is all-False → `num_valid == 0` → loss returns 0 silently, every step.

---

## Root Cause Chain

```
lmbda=0.5 → 50% of steps use student-generated responses (on-policy)
    ↓
Student quality degrades slightly (natural early-training noise)
    ↓
Student starts generating short/empty completions
(max_completion_length=324 is short; EOS token learned early)
    ↓
Empty completions → all labels = -100 → num_valid = 0 → loss = 0.0
    ↓
Zero loss → zero gradient → model parameters freeze
    ↓
Frozen model keeps generating empty completions → loop locked
    ↓
Run resumed from ckpt-10988 (already collapsed)
→ 7000+ more steps of zero-loss "training"
    ↓
Final checkpoints output nothing at inference
```

### Contributing factors

| Factor | Detail |
|--------|--------|
| **Deterministic on-policy schedule** | `_get_random_num()` uses `Random(seed + global_step).random()` — fully deterministic, not truly random. Consecutive steps can form long runs of all-student or all-dataset, amplifying collapse. |
| **Short `max_completion_length=324`** | Student easily hits EOS or gets truncated. Truncated sequences mean fewer valid label tokens, increasing risk of `num_valid=0`. |
| **No collapse detection** | Loss of 0.0 was not caught. Training ran to completion silently. |
| **Resume from collapsed checkpoint** | ckpt-10988 was already collapsed. Resuming from it wasted 7000+ steps. |
| **`group_by_length=true`** | As short student completions accumulate, length-grouped batches increasingly contain short sequences, reinforcing the collapse. |

---

## Diagnostics to Run

To find the **exact collapse step**:
```bash
# Find last step with non-zero loss across all v1 jobs
for f in logs/gkd_tvision_full/gkd_tvision_full_28{1,2,3}*.out; do
    echo "=== $f ==="
    grep "'loss'" "$f" | grep -v "loss': 0.0" | tail -3
done
```

To verify deterministic schedule pattern around the collapse:
```python
import random
for step in range(8000, 11000):
    val = random.Random(step).random()
    src = "STUDENT" if val <= 0.5 else "DATASET"
    print(step, round(val, 3), src)
```

---

## Plan for Next Run

### Changes implemented

1. **Completion length tracking + wandb logging** added to `gkd_trainer.py`:
   - `_logs['completion_len']` deque tracks token lengths of on-policy completions each step
   - Both vLLM and non-vLLM generation paths captured (token ids for vLLM, `labels != -100` count for HF generate)
   - At every logging step, `completion_len/mean`, `completion_len/min`, `completion_len/max` are pushed as wandb scalars
   - `completion_len` column added to the completions Table and `completions.jsonl`
   - **This metric would have caught the collapse at step ~9155**: mean/min lengths would have dropped to 0 several steps before loss did

2. **Early stopping callback** (`ZeroLossEarlyStoppingCallback`) added to `gkd_trainer.py`:
   - Monitors loss at every logging step
   - Stops training if `loss <= 1e-8` for `patience` consecutive logging steps (default: 5)
   - Controllable via `--zero_loss_patience N` arg
   - Would have stopped the run at step ~9185 instead of letting it run to 18311

2. **Increase `max_completion_length`** from 324 → **524** to reduce premature EOS:
   ```bash
   --max_completion_length 524
   ```

### Recommended script changes for next run

3. **Reduce `lmbda`** significantly. Because the student starts from PE only (no instruction tuning),
   early on-policy generations are completely incoherent. `lmbda=0.5` means half of all early steps
   train on garbage. Options:
   - **Conservative:** `--lmbda 0.1` flat throughout
   - **Better:** warm-up `lmbda` from 0 → 0.5 over the first ~30% of training, so the model first
     learns instruction-following from off-policy data before being asked to generate on-policy

4. **Add `min_new_tokens`** to prevent immediate EOS collapse:
   ```bash
   --generation_config '{"min_new_tokens": 10}'
   ```

5. **Start from last known good checkpoint** (ckpt-8241), not the collapsed ckpt-10988:
   ```bash
   --resume_from_checkpoint .../v1-20260316-112802/checkpoint-8241
   ```

6. **Monitor these signals in wandb during training:**
   - `loss` — any sustained drop to 0.0 is a red flag
   - `grad_norm` — constant value (especially `2.44948983`) means params are frozen
   - Student completion lengths (add to `completions.jsonl` logging if possible)

### Gemma 2 + Flash-Attention 2 (NaN risk)

Gemma 2 uses a soft-capping technique on attention logits (tanh capping before softmax).
Standard `flash_attention_2` kernels do not natively support this, which can produce NaN logits
during training, leading to silent loss corruption.

**Current approach:** using `attn_implementation="flash_attention_2"` anyway, since:
- Installing the proper fix (`kernels-community/flash-attn2`) requires significant environment changes
- NaNs are not guaranteed — they tend to surface more with long sequences, large batches, or low precision
- SDPA (`attn_implementation="sdpa"`) avoids the NaN risk but is noticeably slower for Gemma 2 (soft-capping cannot be fused)

**Action:** After this training run, evaluate results. If loss curves look clean and eval scores are
reasonable, flash-attn2 was fine for our configuration. If NaN-related artifacts are observed
(sudden loss spikes, degraded checkpoints with no obvious cause), migrate to `sdpa` or the
community kernel.

**Monitor:** Watch wandb loss for sudden NaN-driven spikes distinct from the on-policy collapse pattern.

---

### TODO (2026-03-19): Fix multi-turn label masking for vLLM on-policy rollouts

**Bug:** In the vLLM on-policy path, `_prepare_batch_inputs(generated_inputs, encode_prompt_only=False)`
encodes in 'train' mode, which labels **all** assistant turns in a multi-turn conversation. But the
student only generated the **last** response — earlier assistant turns should not contribute to the
JSD loss.

The HF-generate path is already correct (`generate_on_policy_outputs` masks everything before the
prompt via `new_labels[:, :prompt_input_ids.shape[1]] = -100`). Only vLLM is affected.

**Fix plan:** After the vLLM `_prepare_batch_inputs` call, mask all labels except the last turn:
```python
labels = encoded_inputs['labels']  # [batch, seq_len]
for i in range(labels.shape[0]):
    mask_pos = (labels[i] == -100).nonzero(as_tuple=True)[0]
    if len(mask_pos) > 0:
        labels[i, :mask_pos[-1] + 1] = -100
```

Also remove the `pdb.set_trace()` debug breakpoint left in `training_step`.

---

### Medium priority (future runs)

7. **Validate first N steps after any resume** before launching full job — confirm loss > 0.

8. **Evaluate intermediate checkpoints automatically** to catch degradation before it compounds.

---

## Last Known Good Checkpoint

```
/e/project1/jureap126/gviveiros/tvision/output/
  gkd_tvision_full_28_beta_0.5_lmbda_0.5_TowerVision-9B/
    v1-20260316-112802/checkpoint-8241
```

Use this as the starting point for any resumed or new run.