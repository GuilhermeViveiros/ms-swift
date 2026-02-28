# 4 * 45GiB, 10.29s/it
# Offline: use models from cache (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)
# Pre-download when online: huggingface-cli download utter-project/TowerVision-2B OpenGVLab/InternVL3-2B-Pretrained
# Or use --model /path/to/cached/TowerVision-2B for explicit local path
#
# IMPORTANT: vLLM and training must run on the SAME node. Use `swift rollout` (NOT deploy)
#   - deploy = inference-only API; rollout = GKD/GRPO endpoints (init_communicator, infer, etc.)
# If using srun:
#   1. srun -A jureap126 -p booster --nodes=1 --time 01:00:00 --pty bash
#   2. CUDA_VISIBLE_DEVICES=0 swift rollout --model utter-project/TowerVision-2B --vllm_engine_kwargs '{"hf_overrides":{"text_config":{"architectures":["Gemma2ForCausalLM"]}}}' --infer_backend vllm --max_new_tokens 2048 &
#   3. until curl -s http://127.0.0.1:8000/health/; do sleep 5; done
#   4. bash examples/train/multimodal/rlhf/gkd/full.sh


datasets=(
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/llava-next-cc-ocr-multi-lan-train.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/dvqa.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/gemini-aokvqa-filtered.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/gemini-chartqa-filtered.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/gemini-docvqa-filtered.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/gemini-iconqa-filtered.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/gemini-infographic-vqa-filtered.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/gemini-rlaif-4v-filtered.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/gemini-textvqa-filtered.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/okvqa.json'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/plotqa.jsonl'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/r1-vision-ai2d.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/r1-vision-scienceqa.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/r1-vision-stratos-17k.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/st_vqa.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/tabmwp.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/tally_qa.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/VisionBlocks-pixmo-ask-model-anything.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/VisionBlocks-pixmo-cap.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/VisionBlocks-pixmo-cap-qa.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/pixmo-count.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/pixmo-docs.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/vqav2.json'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/euroblocks-sft-0525-text-only.jsonl'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/pixmo-cap-translated.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/pangea-cultural-150k.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/pangea-multi-1m.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/Curated-CulturalGround-OE-Filtered-401149.json'
    # '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/Curated-CulturalGround-MCQs-Filtered-379834.json'
)

export HF_HUB_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 

# Option A: Server mode - rollout and training in separate processes (requires same node).
# Start rollout first, wait for health, then run training. NCCL can fail on some HPC clusters.
# Option B: Colocate mode (recommended) - vLLM runs inside the trainer, no NCCL/network needed.
#   Set --vllm_mode colocate and remove the rollout block below.
#
# For server mode, uncomment and run rollout first:
# CUDA_VISIBLE_DEVICES=0 swift rollout --model ... --host 127.0.0.1 --port 8000 > towervision.log 2>&1 &
# until curl -s http://127.0.0.1:8000/health/; do sleep 5; done
#
# CUDA_VISIBLE_DEVICES=0 \
# swift rollout \
#     --model /e/scratch/jureap126/gviveiros/hf_models/TowerVision-2B \
#     --vllm_engine_kwargs '{"hf_overrides":{"text_config":{"architectures":["Gemma2ForCausalLM"]}}}' \
#     --infer_backend vllm \
#     --torch_dtype bfloat16 \
#     --attn_impl flash_attn \
#     --host 127.0.0.1 \
#     --port 8000 \
#     > towervision.log 2>&1 &
    

# OOM: teacher_deepspeed zero3 shards 9B teacher across GPUs. If still OOM: NPROC_PER_NODE=1 or zero3_offload.
# but in vllm colate for some reason I have a bug when I use lash attebntion in the genration process... with the vllm server it does not happen
#LOG_LEVEL=ERROR \
ROOT_IMAGE_DIR=/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/images \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=29501 \
NPROC_PER_NODE=1 \
swift rlhf \
    --rlhf_type gkd \
    --model /e/scratch/jureap126/gviveiros/hf_models/TowerVision-2B \
    --teacher_model /e/scratch/jureap126/gviveiros/hf_models/TowerVision-9B \
    --teacher_deepspeed zero3 \
    --dataset ${datasets[@]} \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --tuner_type full \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --use_hf true \
    --seq_kd false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_engine_kwargs '{"hf_overrides":{"architectures":["LlavaNextForConditionalGeneration"],"text_config":{"architectures":["Gemma2ForCausalLM"]}}}' \
    --vllm_gpu_memory_utilization 0.1 \
    --lmbda 0.5 \
    --beta 0.5 \
    --temperature 0.9 \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 2 \
    --gradient_checkpointing true \
    --logging_steps 2 \
    --max_completion_length 512 \
    --output_dir /e/scratch/jureap126/gviveiros/tvision/output/ \
    --offload_teacher_model true \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 2 \
    --save_only_model true \
    --deepspeed zero2 \
    #--padding_free true
    #--max_length 7000 \