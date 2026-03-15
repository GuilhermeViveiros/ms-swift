#!/bin/bash
#SBATCH --job-name=gkd_tvision_full
#SBATCH --output=logs/%x/%x_%j.out
#SBATCH --error=logs/%x/%x_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=70
#SBATCH --nodes=24
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=booster
#SBATCH --account=jureap126
#SBATCH --time=12:00:00


# TowerVision GKD training via swift rlhf (HuggingFace backend, swift/rlhf_trainers/gkd_trainer.py)
# Uses vllm_mode colocate - vLLM runs inside the trainer, no separate rollout server needed.
# IMPORTANT: vLLM and training must run on the SAME node. Use `swift rollout` (NOT deploy) for server mode.


# ===============================
# Exported Environment Variables
# ===============================
export WANDB_MODE=offline
export WANDB_DIR=/e/project1/jureap126/gviveiros/ms-swift/wandb/
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600   # 10 min — fail fast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1      # surface errors immediately
export NCCL_ASYNC_ERROR_HANDLING=1    # ADD this — the actual NCCL-level flag
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONWARNINGS="ignore::UserWarning"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${SLURM_JOB_ID:-29501}
export MASTER_PORT=$((MASTER_PORT % 65536))
export DS_LOG_LEVEL=debug
if [ "$MASTER_PORT" -lt 1024 ]; then
    export MASTER_PORT=29501
fi
export NNODES=${SLURM_JOB_NUM_NODES:-1} # number of nodes
export TRAIN_NODES=$((NNODES-1)) # number of training nodes
export NPROC_PER_NODE=4 # number of processes per node
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_FILE=/e/scratch/jureap126/gviveiros/tvision/logs/nccl_debug_%h_%p.log
export PYTHONFAULTHANDLER=1
export DATALOADER_DEBUG=1
export TORCH_WORKER_TIMEOUT=120

# ===============================
# Local Environment Variables
# ===============================
OUTPUT_DIR=/e/scratch/jureap126/gviveiros/tvision/
mkdir -p $OUTPUT_DIR/logs/gkd_tvision_full
source /e/home/jusers/viveiros1/jupiter/envs/swift/bin/activate
# Variables of GKD Trainer
beta=0.5 # JSD divergence interpolation coefficient
lmbda=0.5 # On-Policy learning probability

PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE * NNODES))
RUN_NAME=gkd_tvision_full_${NNODES}_beta_${beta}_lmbda_${lmbda}
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
gpus=$(python -c "import torch; print(torch.cuda.device_count())")

echo "SLURM_JOBID: ${SLURM_JOBID}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES: ${NNODES}, NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "GLOBAL_BATCH_SIZE: ${GLOBAL_BATCH_SIZE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}, gpus visible: ${gpus}"
echo "RUN_NAME: ${RUN_NAME}"

cd /e/home/jusers/viveiros1/jupiter/ms-swift

# '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/VisionBlocks-pixmo-cap.jsonl'
datasets=(
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/llava-next-cc-ocr-multi-lan-train.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/dvqa.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/gemini-aokvqa-filtered.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/gemini-chartqa-filtered.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/gemini-docvqa-filtered.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/gemini-iconqa-filtered.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/gemini-infographic-vqa-filtered.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/gemini-rlaif-4v-filtered.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/gemini-textvqa-filtered.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/okvqa.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/plotqa.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/r1-vision-ai2d.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/r1-vision-scienceqa.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/r1-vision-stratos-17k.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/st_vqa.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/tabmwp.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/tally_qa.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/VisionBlocks-pixmo-ask-model-anything.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/VisionBlocks-pixmo-cap-qa.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/pixmo-count.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/pixmo-docs.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/vqav2.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/euroblocks-sft-0525-text-only_filtered.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/pixmo-cap-translated.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/pangea-cultural-150k.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/pangea-multi-1m.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/Curated-CulturalGround-OE-Filtered-401149.jsonl'
    '/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data/Curated-CulturalGround-MCQs-Filtered-379834.jsonl'
)


# ===============================
# Rollout Server
# ===============================
# Wait for rollout address file
# ROLLOUT_DIR=/e/scratch/jureap126/gviveiros/tvision/rollout/${RUN_ID}
# ROLLOUT_ADDR_FILE=/e/scratch/jureap126/gviveiros/tvision/rollout_addr.txt
# #mkdir -p $ROLLOUT_DIR
# #ROLLOUT_ADDR_FILE=${ROLLOUT_DIR}/addr.txt
# echo "Waiting for rollout address file..."
# while [ ! -f $ROLLOUT_ADDR_FILE ]; do sleep 5; done
# ROLLOUT_IP=$(cat $ROLLOUT_ADDR_FILE | tr -d '[:space:]')
# echo "Rollout server IP: $ROLLOUT_IP"

# # Wait for rollout server to be healthy
# echo "Waiting for rollout server to be healthy..."
# until curl -sf http://${ROLLOUT_IP}:8000/health/ > /dev/null 2>&1; do
#     echo "  not ready yet, retrying in 10s..."
#     sleep 10
# done
# echo "Rollout server is ready!"

#--vllm_mode server \
#--vllm_server_host $ROLLOUT_IP \

# ===============================
# Swift Command
# ===============================
SWIFT_CMD="ROOT_IMAGE_DIR=/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/images \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} \
swift rlhf \
    --rlhf_type gkd \
    --model /e/scratch/jureap126/gviveiros/hf_models/TowerVision-2B \
    --teacher_model /e/scratch/jureap126/gviveiros/hf_models/TowerVision-9B \
    --dataset ${datasets[@]} \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --tuner_type full \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --vit_lr 2e-6 \
    --seq_kd false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_engine_kwargs '{"hf_overrides":{"architectures":["LlavaNextForConditionalGeneration"],"text_config":{"architectures":["Gemma2ForCausalLM"]}}}' \
    --vllm_gpu_memory_utilization 0.20 \
    --lmbda 0.5 \
    --beta 0.5 \
    --temperature 0.9 \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_steps 0 \
    --eval_strategy no \
    --save_steps 0.15 \
    --save_total_limit 5 \
    --use_logits_to_keep true \
    --gradient_checkpointing true \
    --logging_steps 5 \
    --save_on_each_node false \
    --max_completion_length 324 \
    --output_dir /e/scratch/jureap126/gviveiros/tvision/output/${RUN_NAME} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --dataset_num_proc 2 \
    --save_only_model false \
    --max_length 5000 \
    --deepspeed zero2 \
    --offload_teacher_model true \
    --teacher_deepspeed zero2 \
    --resume_from_checkpoint /e/scratch/jureap126/gviveiros/tvision/output/gkd_tvision_full_24_beta_0.5_lmbda_0.5_teacher_2b/v1-20260314-185611/checkpoint-17200 \
    --group_by_length true"

# --resume_from_checkpoint /e/scratch/jureap126/gviveiros/tvision/output/gkd_tvision_full_24_beta_0.5_lmbda_0.5_teacher_2b/v1-20260314-185611/checkpoint-17200 \
# --resume_from_checkpoint /e/scratch/jureap126/gviveiros/tvision/output/gkd_tvision_full_24_beta_0.5_lmbda_0.5/v24-20260311-163423/checkpoint-430/ \
# --teacher_deepspeed zero3 \
# --offload_teacher_model true \

# ===============================
# Launch Training
# ===============================
srun --ntasks=$NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 --job-name=${RUN_NAME} \
    bash -c "export NODE_RANK=\${SLURM_PROCID:-0}; export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0,1,2,3}; $SWIFT_CMD"