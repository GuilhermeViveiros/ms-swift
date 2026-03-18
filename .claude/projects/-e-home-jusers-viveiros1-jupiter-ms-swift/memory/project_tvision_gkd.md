---
name: TowerVision GKD Project
description: Research goal and model setup for the TowerVision on-policy distillation project
type: project
---

On-policy GKD distillation experiment to improve multimodal synergy in TowerVision models.

**Research question:** Can on-policy distillation improve synergy between modalities — preserving or increasing text capabilities while increasing vision skills?

**Model setup:**
- Student: TowerVision-2B after projector alignment stage (NOT full SFT) — backbone is pretrained LLM with aligned vision projector
- Teacher: TowerVision-9B final SFT version (fully instruction-tuned)
- Trainer: custom GKD trainer in `swift/rlhf_trainers/gkd_trainer.py` using JSD loss with lmbda on-policy mixing

**Why:** Standard SFT on vision data risks degrading text capabilities. On-policy distillation from the larger SFT teacher should transfer both vision and language knowledge simultaneously, preserving the text-language synergy the 9B model has.

**How to apply:** Frame all suggestions around the dual objective: vision improvement + text capability preservation. Eval should always include multilingual text benchmarks (ALM-Bench) alongside vision benchmarks (OCRBench, etc.).
