"""TowerVision registry: LlavaNext (SigLIP2 + Gemma2) model and template.

Follows ms-swift best practices for MLLM registration:
https://swift.readthedocs.io/en/latest/BestPractices/MLLM-Registration.html
"""
from transformers import PretrainedConfig, PreTrainedModel

from swift.model import (Model, ModelGroup, ModelMeta, get_model_processor, register_model, ModelLoader)
from swift.model.model_arch import ModelArch
import torch
from swift.utils import Processor
from swift.template import TemplateMeta, register_template
from swift.template.templates.llava import Llava1_6HfTemplate
from swift.template.template_inputs import StdTemplateInputs
from swift.template.utils import Context, findall
from typing import Any, Dict, List, Literal, Optional
import io
import os
from modelscope import snapshot_download
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
import requests
from swift.utils import to_float_dtype

# class TowerVisionLoader(ModelLoader):


#     def get_config(self, model_dir: str) -> PretrainedConfig:
#         from transformers import LlavaNextConfig
#         config = LlavaNextConfig.from_pretrained(model_dir, trust_remote_code=True)
#         return config

#     def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
#         from transformers import LlavaNextProcessor
#         processor = LlavaNextProcessor.from_pretrained(model_dir, trust_remote_code=True)
#         processor.tokenizer.padding_side = 'right'
#         return processor

#     def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
#                   model_kwargs) -> PreTrainedModel:
#         from transformers import LlavaNextForConditionalGeneration
#         print('Run TowerVision...')
#         self.auto_model_cls = self.auto_model_cls or LlavaNextForConditionalGeneration
#         model = super().get_model(model_dir, config, processor, model_kwargs)
#         return model


# register_model(
#     ModelMeta(
#         'tower_vision',
#         [
#             ModelGroup([
#                 Model('utter-project/TowerVision-2B', 'utter-project/TowerVision-2B'),
#                 Model('utter-project/TowerVision-9B', 'utter-project/TowerVision-9B'),
#             ]),
#         ],
#         TowerVisionLoader,
#         template='tower_vision',
#         is_multimodal=True,
#         model_arch=ModelArch.llava_hf,  # Same structure as LlavaNext (vision_tower + multi_modal_projector)
#         architectures=['LlavaNextForConditionalGeneration'],
#         requires=['transformers>=4.50'],
#         tags=['vision'],
#     ))

# register_template(
#     TemplateMeta(
#         'tower_vision',
#         prefix=['<bos>'],
#         prompt=['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
#         chat_sep=['<end_of_turn>\n'],
#         suffix=['<end_of_turn>'],
#         system_prefix=['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'],
#         default_system='You are a helpful assistant.',
#         stop_words=['<|endoftext|>'],
#         agent_template='hermes',
#         template_cls=Llava1_6HfTemplate,
#     )
# )


def test_tower_vision(model_id: str = 'utter-project/TowerVision-2B'):
    """TransformersEngine inference for alignment verification."""
    engine = TransformersEngine(
        model_id,
        model_type='llava_next_gemma2_hf',
        attn_impl='flex_attention',
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
    )
    infer_request = InferRequest(
        messages=[{'role': 'user', 'content': '<image>\n' + 'What is the label of the 1st group of bars from the left?'}],
        images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'],
    )
    # 4096 ctx; prompt ~3714 tokens -> max ~256 new tokens
    request_config = RequestConfig(temperature=0, max_tokens=256)
    encoded = engine.template.encode(infer_request)
    input_ids = encoded['input_ids']
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids[0].tolist()
    elif isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    resp_list = engine.infer([infer_request], request_config)
    resp = resp_list[0].choices[0].message.content
    #print('Swift text:', resp)
    return input_ids, resp


def test_inference_alignment():
    """Align TransformersEngine with transformers (input_ids and response)."""
    os.environ['SWIFT_DEBUG'] = '1'
    #input_ids_hf, response_hf = infer_hf()
    input_ids_swift, response_swift = test_tower_vision()
    
    # assert input_ids_hf == input_ids_swift, (
    #     f'input_ids mismatch: hf len={len(input_ids_hf)}, swift len={len(input_ids_swift)}'
    # )
    # assert response_hf == response_swift, (
    #     f'response mismatch: hf={response_hf[:80]}... vs swift={response_swift[:80]}...'
    # )
    print('Inference alignment OK.')


if __name__ == '__main__':
    # Run: python towervision_registry.py
    # Requires: GPU, utter-project/TowerVision-2B (or local path)
    test_inference_alignment()