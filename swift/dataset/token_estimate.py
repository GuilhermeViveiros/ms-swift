# Copyright (c) ModelScope Contributors. All rights reserved.
"""Token estimation for multimodal rows (text + images) to filter by max_length."""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from swift.template import MaxLengthError
if TYPE_CHECKING:
    from swift.template import Template


def _get_image_size(path_or_bytes: Any) -> Optional[Tuple[int, int]]:
    """Get (height, width) from image path, bytes, or dict with path/bytes. Returns None on failure."""
    try:
        from PIL import Image
        from swift.template.vision_utils import load_file

        if isinstance(path_or_bytes, dict):
            path_or_bytes = path_or_bytes.get('bytes') or path_or_bytes.get('path')
        if path_or_bytes is None:
            return None
        f = load_file(path_or_bytes)
        if hasattr(f, 'read'):
            img = Image.open(f)
        else:
            img = Image.open(path_or_bytes)
        w, h = img.size  # PIL returns (width, height)
        return (h, w)  # return (height, width) for processor
    except Exception:
        return None


def _get_image_sizes_from_row(row: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Extract image paths from row and return list of (height, width) per image."""
    sizes = []
    images = row.get('images') or []
    if isinstance(images, str):
        images = [images]
    for img in images:
        path = img.get('path') if isinstance(img, dict) else img
        if path is None and isinstance(img, dict) and img.get('bytes'):
            path = img
        if path is not None:
            sz = _get_image_size(path)
            if sz:
                sizes.append(sz)
    # Also from messages content (e.g. {"type": "image", "image": {"url": "..."}})
    for msg in row.get('messages', []):
        content = msg.get('content')
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            key = item.get('type', '')
            if key.endswith('_url'):
                key = key[:-4]
            if key in ('image', 'audio', 'video'):
                val = item.get(key) or item.get(f'{key}_url')
                if isinstance(val, dict):
                    val = val.get('url')
                if val:
                    sz = _get_image_size(val) if key == 'image' else None
                    if sz:
                        sizes.append(sz)
    return sizes


def _estimate_image_tokens_llava_next(
    processor: Any,
    image_sizes: List[Tuple[int, int]],
) -> int:
    """
    Estimate image tokens using the same logic as LlavaHfTemplate._encode.
    Uses processor._get_number_of_features with processed size from image_processor.
    Applies vision_feature_select_strategy == 'default' subtract 1.
    """
    if not image_sizes or not hasattr(processor, '_get_number_of_features'):
        return 0
    # Get processed size from image_processor (same as LlavaHfTemplate)
    ip = getattr(processor, 'image_processor', None)
    proc_h = proc_w = 384
    total = 0
    for orig_h, orig_w in image_sizes:
        n = processor._get_number_of_features(orig_h, orig_w, 384, 384)
        if getattr(processor, 'vision_feature_select_strategy', None) == 'default':
            n -= 1
        total += n
    return total


def ensure_vision_tokens_in_row(row: Dict[str, Any]) -> bool:
    """
    1. For image samples, ensure the number of images = 1 and the vision tokens = nb of images.
    """
    images = row.get('images', [])
    messages = row.get('messages', [])
    vision_token_count = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            vision_token_count += content.count("<image>")
    if len(images) > 1:
        return False
    if len(images) != vision_token_count:
        return False
    return True

from typing import Any

_processor_cache = {}

def load_processor(model_dir: str) -> Any:
    """
    Loads a LlavaNextProcessor from transformers, caching by model_dir.
    """
    if model_dir in _processor_cache:
        return _processor_cache[model_dir]
    from transformers import LlavaNextProcessor
    proc = LlavaNextProcessor.from_pretrained(model_dir, trust_remote_code=True)
    _processor_cache[model_dir] = proc
    return proc

def estimate_tokens_for_mllm_row(
    row: Dict[str, Any],
    processor: Any,
) -> int:
    """
    Estimate total token count for a multimodal row.

    Args:
        row: Dataset row with 'messages' and optionally 'images'.
        processor: Processor.

    Returns:
        Total token count, or -1 if invalid/cannot estimate.
    """
    if 'messages' not in row:
        return -1

    # if not use_lightweight:
    #     # Accurate path: use the same template.encode as training
    #     try:
    #         encoded = template.encode(row, return_length=True)
    #         lengths = encoded.get('lengths', [len(encoded.get('input_ids', []))])
    #         total = max(lengths) if isinstance(lengths, list) else lengths
    #         return int(total)
    #     except Exception as e:
    #         return -1

    # Lightweight path: use template.processor (same as template uses during encode)
    # proc = load_processor("/e/scratch/jureap126/gviveiros/hf_models/TowerVision-2B")
    
    if row.get('images') is not None:
        image_sizes = _get_image_sizes_from_row(row)
        img_tokens = _estimate_image_tokens_llava_next(processor, image_sizes)
    else:
        img_tokens = 0

    

    text_parts = []
    for msg in row.get('messages', []):
        c = msg.get('content', '')
        if isinstance(c, str):
            text_parts.append(c)
        elif isinstance(c, list):
            for it in c:
                if isinstance(it, dict) and it.get('type') == 'text':
                    text_parts.append(it.get('text', ''))
    text = ' '.join(text_parts) or ''
    text_tokens = len(processor.tokenizer.encode(text, add_special_tokens=False))

    template_overhead = 128
    total = img_tokens + text_tokens + template_overhead
    return img_tokens, int(total)
