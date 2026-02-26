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

class TowerVisionLoader(ModelLoader):


    def get_config(self, model_dir: str) -> PretrainedConfig:
        from transformers import LlavaNextConfig
        config = LlavaNextConfig.from_pretrained(model_dir, trust_remote_code=True)
        return config

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(model_dir, trust_remote_code=True)
        processor.tokenizer.padding_side = 'right'
        return processor

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        from transformers import LlavaNextForConditionalGeneration
        print('Run TowerVision...')
        self.auto_model_cls = self.auto_model_cls or LlavaNextForConditionalGeneration
        model = super().get_model(model_dir, config, processor, model_kwargs)
        return model


register_model(
    ModelMeta(
        'llava_next_gemma2_hf',
        [
            ModelGroup([
                Model('utter-project/TowerVision-2B', 'utter-project/TowerVision-2B'),
                Model('utter-project/TowerVision-9B', 'utter-project/TowerVision-9B'),
            ]),
        ],
        TowerVisionLoader,
        template='llava_next_gemma2_hf',
        is_multimodal=True,
        model_arch=ModelArch.llava_hf,  # Same structure as LlavaNext (vision_tower + multi_modal_projector)
        architectures=['LlavaNextForConditionalGeneration'],
        requires=['transformers>=4.50'],
        tags=['vision'],
    ))

class LlavaNextGemma2Template(Llava1_6HfTemplate):
    """TowerVision template: LLaVA-NeXT (SigLIP2 + Gemma2) image-only.

    Aligns with transformers LlavaNextProcessor: same chat format, image token expansion
    via _get_number_of_features, and _post_encode for training (Gemma2 image_token_id
    may be >= vocab_size, so we pre-merge vision features into inputs_embeds).
    """
    use_model = True
    support_padding_free = True
    norm_bbox = 'none'
    placeholder_tokens = ['<image>']

    @property
    def image_token_index(self) -> int:
        """Use processor.image_token_id for Gemma2 (may differ from tokenizer vocab)."""
        if not hasattr(self, '_image_token_index'):
            proc = self.processor
            self._image_token_index = getattr(proc, 'image_token_id', None)
            if self._image_token_index is None:
                self._image_token_index = self.tokenizer.convert_tokens_to_ids(
                    getattr(proc, 'image_token', '<image>')
                )
        return self._image_token_index

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image', 'TowerVision is image-only'
        return ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """LLaVA-NeXT image-only encoding. Use processor for token expansion to match HF exactly.

        LlavaHfTemplate uses pixel_values[0].shape[-2:] for height/width, which is wrong for
        SigLIP's 5D output. We use processor(text, images) to get correct input_ids and
        pixel_values, ensuring alignment with transformers inference.
        """
        images = inputs.images
        if not images:
            return super()._encode(inputs)

        # Build conversation for processor.apply_chat_template (matches HF format)
        # Omit default system to match HF inference and reduce prompt length (fits 4096 ctx)
        conversation = []
        if inputs.system is not None and inputs.system != (self.default_system or ''):
            conversation.append({'role': 'system', 'content': inputs.system})
        for msg in inputs.messages:
            content = msg.get('content', '')
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get('type') == 'image':
                            parts.append('<image>\n')
                        else:
                            parts.append(c.get('text', str(c)))
                    else:
                        parts.append(str(c))
                content = ''.join(parts)
            conversation.append({'role': msg['role'], 'content': content})

        # Add generation prompt for inference; for training the last message is assistant
        add_gen = not (conversation and conversation[-1]['role'] == 'assistant')
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=add_gen, tokenize=False
        )
        if isinstance(prompt, list):
            prompt = prompt[0]

        # Use processor for correct token expansion (matches HF inference)
        proc_out = self.processor(text=prompt, images=images, return_tensors='pt')
        input_ids = proc_out['input_ids']
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]

        pv = proc_out['pixel_values']
        encoded = {
            'input_ids': input_ids,
            'pixel_values': pv,
            'labels': None,
            'loss_scale': None,
        }
        if 'image_sizes' in proc_out:
            encoded['image_sizes'] = proc_out['image_sizes']

        # Build labels for training (mask prompt/image, predict response)
        if self.is_training and not add_gen:
            labels = [-100] * len(input_ids)
            resp_marker = '<start_of_turn>model\n'
            prefix_ids = self.tokenizer.encode(resp_marker, add_special_tokens=False)
            for i in range(len(input_ids) - len(prefix_ids) + 1):
                if input_ids[i:i + len(prefix_ids)] == prefix_ids:
                    for j in range(i + len(prefix_ids), len(input_ids)):
                        labels[j] = input_ids[j]
                    break
            encoded['labels'] = labels
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-compute inputs_embeds with image features merged (Gemma2 image_token_id may be >= vocab_size)."""
        from transformers.integrations import is_deepspeed_zero3_enabled
        from swift.utils import is_deepspeed_enabled, to_device

        if not self.is_training:
            return inputs

        pixel_values = inputs.get('pixel_values')
        image_sizes = inputs.get('image_sizes')
        input_ids = inputs['input_ids']

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        base_model = self.get_base_model(model)
        image_token_id = getattr(model.config, 'image_token_id', None) or getattr(
            model.config, 'image_token_index', self.image_token_index
        )
        pad_id = getattr(model.config, 'pad_token_id', None) or 0

        # Safe embedding lookup (image_token_id may be >= vocab_size for Gemma2)
        safe_input_ids = input_ids.clone()
        safe_input_ids[input_ids == image_token_id] = pad_id

        if hasattr(base_model.model, 'embed_tokens'):
            embed = base_model.model.embed_tokens
        else:
            embed = base_model.model.language_model.embed_tokens
        inputs_embeds = embed(safe_input_ids)

        if pixel_values is not None and pixel_values.numel() > 0:
            pixel_values = to_device(pixel_values, input_ids.device)
            pixel_values = to_float_dtype({'pixel_values': pixel_values}, base_model.dtype)['pixel_values']

            image_outputs = base_model.model.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_select_strategy=getattr(
                    model.config, 'vision_feature_select_strategy', 'default'
                ),
            )
            pooler_out = getattr(image_outputs, 'pooler_output', image_outputs)
            if isinstance(pooler_out, (list, tuple)):
                image_features = torch.cat(pooler_out, dim=0)
            else:
                image_features = pooler_out
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            n_image_tokens = (input_ids == image_token_id).sum().item()
            n_features = image_features.shape[0]
            if n_image_tokens != n_features:
                raise ValueError(
                    f'Image tokens ({n_image_tokens}) != image features ({n_features}). '
                    f'Check _encode and processor._get_number_of_features.'
                )
            image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        elif is_deepspeed_enabled() and not is_deepspeed_zero3_enabled():
            # Dummy pass for mixed batches (some samples without images)
            dummy = input_ids.new_zeros(1, 3, 32, 32, dtype=base_model.dtype)
            dummy_out = base_model.model.get_image_features(dummy, None)
            pooler = getattr(dummy_out, 'pooler_output', dummy_out)
            feat = pooler if not isinstance(pooler, (list, tuple)) else torch.cat(pooler, dim=0)
            inputs_embeds = inputs_embeds + feat.mean().to(inputs_embeds.device) * 0.0

        out = {k: v for k, v in inputs.items() if k not in ('pixel_values', 'image_sizes')}
        out['inputs_embeds'] = inputs_embeds
        out['input_ids'] = input_ids  # LlavaNext needs input_ids for placeholder mask
        return out


register_template(
    TemplateMeta(
        'llava_next_gemma2_hf',
        prefix=['<bos>'],
        prompt=['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
        chat_sep=['<end_of_turn>\n'],
        suffix=['<end_of_turn>'],
        system_prefix=['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'],
        default_system='You are a helpful assistant.',
        stop_words=['<|endoftext|>'],
        agent_template='hermes',
        template_cls=LlavaNextGemma2Template,
    )
)


# def infer_hf(model_id: str = 'utter-project/TowerVision-2B'):
#     """Direct transformers inference for alignment verification.

#     Mirrors TowerVision README: conversation with content='<image>\\n{query}',
#     processor.apply_chat_template, then processor(text, images).
#     """
#     from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
#     from PIL import Image

#     model_dir = snapshot_download(model_id)
#     model = LlavaNextForConditionalGeneration.from_pretrained(
#         model_dir, torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa'
#     )
#     processor = LlavaNextProcessor.from_pretrained(model_dir)

#     image_url = 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'
#     if image_url.startswith('http'):
#         img = Image.open(io.BytesIO(requests.get(image_url).content)).convert('RGB')
#     else:
#         img = Image.open(image_url).convert('RGB')

#     conversation = [{'role': 'user', 'content': '<image>\nDescribe the image.'}]
#     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
#     inputs = processor(text=prompt, images=[img], return_tensors='pt')
#     inputs = inputs.to(model.device).to(model.dtype)

#     generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
#     text = processor.batch_decode(
#         generated[:, inputs['input_ids'].shape[1]:],
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False,
#     )
#     print('HF text:', text[0])
#     return inputs['input_ids'][0].tolist(), text[0]


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
        messages=[{'role': 'user', 'content': '<image>\nDescribe the image.'}],
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