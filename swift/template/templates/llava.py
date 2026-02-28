# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import transformers
from dataclasses import dataclass, field
from packaging import version
from typing import Any, Dict, List, Literal, Optional

from swift.utils import (
    get_env_args,
    is_deepspeed_enabled,
    to_device,
    to_float_dtype
)
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_video_llava
from .llama import Llama3TemplateMeta
from .gemma import GemmaTemplateMeta
from .qwen import QwenTemplateMeta
from .utils import ChatmlTemplateMeta


class LlavaHfTemplate(Template):
    placeholder_tokens = ['<image>']

    @property
    def image_token_index(self):
        if not hasattr(self, '_image_token_index'):
            self._image_token_index = self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        return self._image_token_index

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        if images:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                encoded['image_sizes'] = image_inputs['image_sizes']
            if version.parse(transformers.__version__) >= version.parse('4.47'):
                input_ids = encoded['input_ids']
                labels = encoded['labels']
                idx_list = findall(input_ids, self.image_token_index)  # <image>
                height, width = image_inputs['pixel_values'][0].shape[-2:]
                added_tokens_len = 0
                for i, idx in enumerate(idx_list):
                    if 'image_sizes' in image_inputs:
                        orig_height, orig_width = image_inputs['image_sizes'][i].tolist()
                        num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height,
                                                                                  width)
                    else:
                        num_image_tokens = (height // self.processor.patch_size) * (
                            width // self.processor.patch_size) + self.processor.num_additional_image_tokens
                    if self.processor.vision_feature_select_strategy == 'default':
                        num_image_tokens -= 1
                    input_ids = input_ids[:added_tokens_len + idx] + [self.image_token_index] * num_image_tokens \
                        + input_ids[added_tokens_len + idx + 1:]
                    if labels is not None:
                        labels = labels[:added_tokens_len + idx] + [-100] * num_image_tokens \
                            + labels[added_tokens_len + idx + 1:]
                    added_tokens_len += num_image_tokens - 1
                encoded['input_ids'] = input_ids
                encoded['labels'] = labels
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.llava1_5_hf,
        prefix=['<s>'],
        prompt=['USER: {{QUERY}}\nASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        system_prefix=['<s>{{SYSTEM}}\n'],
        template_cls=LlavaHfTemplate,
    ))


class LlavaVideoHfTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<image>\n']
        assert media_type == 'video'
        media_file = inputs.videos[index]
        if media_file.rsplit('.', 1)[-1] in {'jpg', 'png'}:
            return ['<image>\n']
        else:
            inputs.videos[index] = load_video_llava(inputs.videos[index])
            return ['<video>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        videos = inputs.videos or []
        if len(videos) > 0:
            video_processor = self.processor.video_processor
            video_inputs = video_processor(videos, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values_videos'] = video_inputs['pixel_values_videos']
        if len(images) > 0:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['image_sizes'] = image_inputs['image_sizes']
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.llava_next_video_hf,
        prefix=['{{SYSTEM}} '],
        prompt=['USER: {{QUERY}} ASSISTANT:'],
        chat_sep=[' '],
        suffix=[['eos_token_id']],
        template_cls=LlavaVideoHfTemplate,
        auto_add_bos=True,
    ))


class Llava1_6HfTemplate(LlavaHfTemplate):

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        for b in batch:
            pixel_values = b.get('pixel_values')
            # if pixel_values is not None:
            #     b['pixel_values'] = pixel_values.squeeze(0)  # 5d -> 4d
        res = super()._data_collator(batch, padding_to=padding_to)
        return res


@dataclass
class LlavaMistralTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<s>[INST] '])
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}} [/INST]'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['</s>[INST] '])
    suffix: Prompt = field(default_factory=lambda: ['</s>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])


register_template(LlavaMistralTemplateMeta(MLLMTemplateType.llava1_6_mistral_hf, template_cls=Llava1_6HfTemplate))

register_template(
    TemplateMeta(
        MLLMTemplateType.llava1_6_vicuna_hf,
        prefix=['<s>'],
        prompt=['USER: {{QUERY}} ASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        default_system=('A chat between a curious human and an artificial intelligence assistant. '
                        "The assistant gives helpful, detailed, and polite answers to the human's questions."),
        system_prefix=['<s>{{SYSTEM}} '],
        template_cls=Llava1_6HfTemplate))


class LLava1_6YiHfTemplate(Llava1_6HfTemplate):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            return [[64000], '\n']
        else:
            return super().replace_tag(media_type, index, inputs)


register_template(ChatmlTemplateMeta(
    MLLMTemplateType.llava1_6_yi_hf,
    template_cls=LLava1_6YiHfTemplate,
))

register_template(Llama3TemplateMeta(
    MLLMTemplateType.llama3_llava_next_hf,
    template_cls=Llava1_6HfTemplate,
))

register_template(QwenTemplateMeta(MLLMTemplateType.llava_next_qwen_hf, template_cls=Llava1_6HfTemplate))


class LlavaNextGemma2Template(Llava1_6HfTemplate):
    """TowerVision template: LLaVA-NeXT (SigLIP2 + Gemma2) image-only.

    Aligns with transformers LlavaNextProcessor: same chat format, image token expansion
    via processor, and _post_encode for training (Gemma2 image_token_id may be >= vocab_size).
    """

    use_model = True
    # vLLM expects raw images; passing pre-processed SigLIP 5D pixel_values causes
    # "values outside [0,1]" because vLLM's processor re-processes them as raw images.
    use_custom_encode_for_vllm = False
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

    # def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
    #     """LLaVA-NeXT image-only encoding. Use processor for token expansion to match HF exactly.

    #     LlavaHfTemplate uses pixel_values[0].shape[-2:] for height/width, which is wrong for
    #     SigLIP's 5D output. We use processor(text, images) to get correct input_ids and
    #     pixel_values, ensuring alignment with transformers inference.
    #     """
    #     images = inputs.images
    #     if not images:
    #         return super()._encode(inputs)

    #     # Build conversation for processor.apply_chat_template (matches HF format)
    #     # Omit default system to match HF inference and reduce prompt length (fits 4096 ctx)
    #     conversation = []
    #     if inputs.system is not None and inputs.system != (self.template_meta.default_system or ''):
    #         conversation.append({'role': 'system', 'content': inputs.system})
    #     for msg in inputs.messages:
    #         content = msg.get('content', '')
    #         if isinstance(content, list):
    #             parts = []
    #             for c in content:
    #                 if isinstance(c, dict):
    #                     if c.get('type') == 'image':
    #                         parts.append('<image>\n')
    #                     else:
    #                         parts.append(c.get('text', str(c)))
    #                 else:
    #                     parts.append(str(c))
    #             content = ''.join(parts)
    #         conversation.append({'role': msg['role'], 'content': content})

    #     # Add generation prompt for inference; for training the last message is assistant
    #     add_gen = not (conversation and conversation[-1]['role'] == 'assistant')
    #     prompt = self.processor.apply_chat_template(
    #         conversation, add_generation_prompt=add_gen, tokenize=False
    #     )
    #     if isinstance(prompt, list):
    #         prompt = prompt[0]

    #     # Use processor for correct token expansion (matches HF inference)
    #     # import pdb; pdb.set_trace()
    #     # create PIL image 2000x2000
    #     # from PIL import Image
    #     # image = Image.new('RGB', (2000, 2000), color='red')
    #     proc_out = self.processor(text=prompt, images=images, return_tensors='pt')
    #     input_ids = proc_out['input_ids']
    #     if isinstance(input_ids, torch.Tensor):
    #         input_ids = input_ids.tolist()
    #     if input_ids and isinstance(input_ids[0], list):
    #         input_ids = input_ids[0]

    #     pv = proc_out['pixel_values']
    #     encoded = {
    #         'input_ids': input_ids,
    #         'pixel_values': pv,
    #         'labels': None,
    #         'loss_scale': None,
    #     }
    #     if 'image_sizes' in proc_out:
    #         encoded['image_sizes'] = proc_out['image_sizes']

    #     # Build labels for training (mask prompt/image, predict response)
    #     if self.is_training and not add_gen:
    #         labels = [-100] * len(input_ids)
    #         resp_marker = '<start_of_turn>model\n'
    #         prefix_ids = self.tokenizer.encode(resp_marker, add_special_tokens=False)
    #         for i in range(len(input_ids) - len(prefix_ids) + 1):
    #             if input_ids[i:i + len(prefix_ids)] == prefix_ids:
    #                 for j in range(i + len(prefix_ids), len(input_ids)):
    #                     labels[j] = input_ids[j]
    #                 break
    #         encoded['labels'] = labels
    #     return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-compute inputs_embeds with image features merged (Gemma2 image_token_id may be >= vocab_size)."""
        #from transformers.integrations import is_deepspeed_zero3_enabled
        

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

        else: #is_deepspeed_enabled() and not is_deepspeed_zero3_enabled():
            # Dummy pass for mixed batches (some samples without images)
            size_ = self.config.vision_config.image_size
            dummy = input_ids.new_zeros(1, 3, size_, size_, dtype=base_model.dtype)
            image_features = torch.zeros(1, 784, self.config.vision_config.hidden_size).to(inputs_embeds.device, inputs_embeds.dtype)

        #     
     

        out = {k: v for k, v in inputs.items() if k not in ('pixel_values', 'image_sizes')}
        out['inputs_embeds'] = inputs_embeds
        out['input_ids'] = input_ids  # LlavaNext needs input_ids for placeholder mask
        return out


register_template(
    TemplateMeta(
        MLLMTemplateType.llava_next_gemma2_hf,
        prefix=['<bos>'],
        prompt=['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
        chat_sep=['<end_of_turn>\n'],
        suffix=['<end_of_turn>'],
        system_prefix=['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'],
        default_system='You are a helpful assistant.',
        stop_words=['<|endoftext|>', '<end_of_turn>\n', '<end_of_turn>'],
        agent_template='hermes',
        template_cls=LlavaNextGemma2Template,
    ))


class LlavaOneVisionHfTemplate(Llava1_6HfTemplate):

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, 151646)  # <image>
        processor = self.processor
        if images:
            image_processor = processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            height, width = image_inputs['pixel_values'][0].shape[-2:]
            added_tokens_len = 0
            for idx, pixel_v, image_size in zip(idx_list, image_inputs['pixel_values'], image_inputs['image_sizes']):
                if isinstance(image_size, torch.Tensor):
                    image_size = image_size.tolist()
                orig_height, orig_width = image_size
                num_image_tokens = processor._get_number_of_features(orig_height, orig_width, height, width)
                input_ids = input_ids[:added_tokens_len
                                      + idx] + [151646] * num_image_tokens + input_ids[added_tokens_len + idx + 1:]
                if labels is not None:
                    labels = labels[:added_tokens_len + idx] + [-100] * num_image_tokens + labels[added_tokens_len + idx
                                                                                                  + 1:]
                added_tokens_len += num_image_tokens - 1
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                encoded['image_sizes'] = image_inputs['image_sizes']
        return encoded


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.llava_onevision_hf,
        default_system=None,
        template_cls=LlavaOneVisionHfTemplate,
    ))


class LlavaLlama3_1HfTemplate(LlavaHfTemplate):
    # DaozeZhang
    system = ('You are a helpful language and vision assistant. '
              'You are able to understand the visual content that the user provides, '
              'and assist the user with a variety of tasks using natural language.')

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if len(encoded['pixel_values'].shape) == 5:  # (1, num_patch, 3, H/W, W/H)
            encoded['pixel_values'] = torch.squeeze(encoded['pixel_values'], dim=0)  # (num_patch, 3, H/W, W/H)
        return encoded


register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llava_llama3_1_hf,
        default_system=LlavaLlama3_1HfTemplate.system,
        template_cls=LlavaLlama3_1HfTemplate,
    ))


class LLavaLlama3HfTemplate(Template):
    # xtuner
    image_placeholder = ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        raw_image = inputs.images
        if raw_image:
            pixel_values = self.processor.image_processor(raw_image, return_tensors='pt')['pixel_values']
            encoded['pixel_values'] = pixel_values.to(self.model_info.torch_dtype)
        return encoded


register_template(Llama3TemplateMeta(
    MLLMTemplateType.llava_llama3_hf,
    template_cls=LLavaLlama3HfTemplate,
))


class LLavaTemplate(Template):
    skip_prompt = False
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        image_sizes = [x.size for x in images]
        from llava.mm_utils import process_images
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        if images:
            images_tensor = process_images(images, image_processor, model.config)
            encoded['images'] = images_tensor.to(model.dtype).squeeze(0)
            encoded['image_sizes'] = image_sizes
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = images
            res['image_sizes'] = sum([b['image_sizes'] for b in batch if 'image_sizes' in b], start=[])
        return res


register_template(LlavaMistralTemplateMeta(MLLMTemplateType.llava1_6_mistral, template_cls=LLavaTemplate))

register_template(ChatmlTemplateMeta(MLLMTemplateType.llava1_6_yi, template_cls=LLavaTemplate))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llama3_llava_next,
        template_cls=LLavaTemplate,
        default_system=('You are a helpful language and vision assistant. '
                        'You are able to understand the visual content that the user provides, '
                        'and assist the user with a variety of tasks using natural language.'),
    ))

register_template(QwenTemplateMeta(MLLMTemplateType.llava_next_qwen, template_cls=LLavaTemplate))


class LLavaOneVision1_5Template(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    use_model = True
    support_padding_free = True

    def init_env_args(self):
        super().init_env_args()
        self.bbox_format = get_env_args('QWENVL_BBOX_FORMAT', str, 'legacy')

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            video, video_kwargs = fetch_video({'video': video}, return_video_sample_fps=True)
            inputs.mm_processor_kwargs.setdefault('fps', []).append(video_kwargs)
            tokens = ['<|vision_start|><|video_pad|><|vision_end|>']
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return tokens

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        else:
            return [ref]

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        else:
            return [str(bbox)]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    kwargs = {}
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                        kwargs['images'] = None
                    media_inputs = processor_func(videos=mm_data, return_tensors='pt', do_resize=False, **kwargs)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
        return {'inputs_embeds': inputs_embeds}


register_template(QwenTemplateMeta(MLLMTemplateType.llava_onevision1_5, template_cls=LLavaOneVision1_5Template))
