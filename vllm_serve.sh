#CUDA_VISIBLE_DEVICES=0 \
# swift deploy \
#     --model utter-project/TowerVision-2B \
#     --vllm_engine_kwargs '{"hf_overrides":{"text_config":{"architectures":["Gemma2ForCausalLM"]}}}' \
#     --infer_backend vllm \
#     --temperature 0 \
#     --max_new_tokens 2048

# For standalone rollout (no trainer), use load_format='auto' so vLLM loads actual weights.
# With default load_format='dummy', rollout expects weights from the trainer and produces garbage.
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model utter-project/TowerVision-2B \
    --vllm_engine_kwargs '{"load_format":"auto","hf_overrides":{"text_config":{"architectures":["Gemma2ForCausalLM"]}}}' \
    --infer_backend vllm \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --temperature 0.9 \
    --max_new_tokens 2048 \
    --host 127.0.0.1 \
    --port 8000


# --- Run these in another terminal after the server is up (rollout blocks above) ---
# Test rollout server (no /v1/chat/completions - that's deploy only)
# 1. Health check:
curl -s http://localhost:8000/health/
# 2. World size (should return {"world_size": 1}):
curl -s http://localhost:8000/get_world_size/
# 3. POST to /infer/ (RolloutInferRequest format) - text-only:
curl -s -X POST http://localhost:8000/infer/ -H "Content-Type: application/json" \
  -d '{"infer_requests":[{"messages":[{"role":"user","content":"Are you good at answering questions?"}]}],"request_config":{"max_tokens":512, "temperature":0}}'
# 4. POST with image (run from ms-swift dir; use absolute path if needed):
#    Use EITHER structured content (type:image) OR top-level images, not both.
curl -s -X POST http://localhost:8000/infer/ -H "Content-Type: application/json" \
  -d '{"infer_requests":[{"messages":[{"role":"user","content":[{"type":"image","image":"/e/home/jusers/viveiros1/jupiter/ms-swift/asset/banner.png"},{"type":"text","text":"Describe the image briefly."}]}]}],"request_config":{"max_tokens":512,"temperature":0}}'
  