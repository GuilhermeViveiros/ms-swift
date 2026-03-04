# Copyright (c) ModelScope Contributors. All rights reserved.
"""Token estimation for multimodal rows (text + images) to filter by max_length."""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# TowerVision (utter-project/TowerVision-2B) uses LlavaNextProcessor
TOWERVISION_MODEL_ID = 'utter-project/TowerVision-2B'

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


def _estimate_text_tokens(row: Dict[str, Any], tokenizer: Any) -> int:
    """Estimate text tokens from messages (no images)."""
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
    return len(tokenizer.encode(text, add_special_tokens=True))


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


def estimate_num_image_tokens_llava_next(
    processor: Any,
    image_sizes: List[Tuple[int, int]],
) -> int:
    """
    Estimate total image tokens for LlavaNext/SigLIP models from image dimensions.

    Uses processor._get_num_multimodal_tokens when available (transformers LlavaNextProcessor).
    Falls back to processor._get_number_of_features per image if needed.

    Args:
        processor: LlavaNextProcessor (or compatible) with image_processor, patch_size, etc.
        image_sizes: List of (height, width) per image.

    Returns:
        Total number of image tokens.
    """
    if not image_sizes:
        return 0
    # Prefer _get_num_multimodal_tokens (batch API)
    if hasattr(processor, '_get_num_multimodal_tokens'):
        mm_data = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        if hasattr(mm_data, 'num_image_tokens'):
            return sum(mm_data.num_image_tokens)
    # Fallback: _get_number_of_features per image (needs processed height/width)
    if hasattr(processor, '_get_number_of_features'):
        ip = getattr(processor, 'image_processor', None)
        if ip is not None and hasattr(ip, 'size'):
            size = getattr(ip, 'size', None) or {}
            if isinstance(size, dict):
                if 'shortest_edge' in size:
                    se = size['shortest_edge']
                    proc_h, proc_w = se, se
                else:
                    h = size.get('height', 384)
                    w = size.get('width', 384)
                    proc_h = proc_w = min(h, w)
            else:
                proc_h = proc_w = 384
        else:
            proc_h = proc_w = 384
        total = 0
        for orig_h, orig_w in image_sizes:
            n = processor._get_number_of_features(orig_h, orig_w, proc_h, proc_w)
            if getattr(processor, 'vision_feature_select_strategy', None) == 'default':
                n -= 1
            total += n
        return total
    return 0


_towervision_processor_cache: Dict[str, Any] = {}


def _get_towervision_processor(model_dir: Optional[str] = None) -> Optional[Any]:
    """Load LlavaNextProcessor for TowerVision. Uses model_dir or default utter-project/TowerVision-2B.
    Cached per model_dir to avoid repeated loads during dataset filtering."""
    path = model_dir or TOWERVISION_MODEL_ID
    if path not in _towervision_processor_cache:
        try:
            from transformers import LlavaNextProcessor
            _towervision_processor_cache[path] = LlavaNextProcessor.from_pretrained(
                path, trust_remote_code=True
            )
        except Exception:
            _towervision_processor_cache[path] = None
    return _towervision_processor_cache[path]


def ensure_vision_tokens_in_row(row: Dict[str, Any]) -> bool:
    """
    1. For image samples, ensure the number of images = 1 and the vision tokens = nb of images.
    """
    images = row.get('images', [])
    messages = row.get('messages', [])
    # search the vision token <image>
    # Count number of <image> tokens across all messages
    vision_token_count = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            vision_token_count += content.count("<image>")

    #num_images = len(images)
    if len(images) > 1:
        return False
    if len(images) != vision_token_count:
        return False
    return True

def estimate_tokens_for_mllm_row(
    row: Dict[str, Any],
    template: 'Template',
    max_length: int,
    *,
    use_lightweight: bool = True,
    model_dir: Optional[str] = None,
) -> bool:
    """
    Estimate whether a multimodal row fits within max_length.

    Returns True if the row should be kept (within limit), False to filter out.

    Args:
        row: Dataset row with 'messages' and optionally 'images'.
        template: Initialized template (processor must be inited for full encode path).
        max_length: Maximum allowed sequence length.
        use_lightweight: If True, use fast image-token estimation for TowerVision when
            possible; otherwise always use full template.encode (slower but accurate).
        model_dir: Model path for TowerVision (e.g. utter-project/TowerVision-2B or
            local path). Used only for lightweight path; processor is loaded directly.

    Returns:
        True if row fits, False to filter out.
    """
    from swift.template import MaxLengthError

    if 'messages' not in row:
        return False

    # TowerVision: img_tokens + text_tokens (processor loaded directly)
    proc = _get_towervision_processor(model_dir)
    if proc is not None and (
        hasattr(proc, '_get_num_multimodal_tokens') or hasattr(proc, '_get_number_of_features')
    ):
        image_sizes = _get_image_sizes_from_row(row)
        img_tokens = 0
        if len(image_sizes) > 0:
            image_sizes = image_sizes[0]
            # count 1 only, if > 1 it will be filtered out in later steps
            img_tokens = proc._get_number_of_features(image_sizes[0], image_sizes[1], 384, 384)
        tokenizer = getattr(template, 'tokenizer', None) or getattr(proc, 'tokenizer', None)
        text_tokens = _estimate_text_tokens(row, tokenizer) if tokenizer else 0
        # Add buffer for template overhead (prefix, suffix, chat format)
        safe_buffer = 328
        total = img_tokens + text_tokens + safe_buffer
    else:
        raise ValueError('TowerVision processor is not loaded or is not compatible with the template.')
    #if total > max_length:
    #print(f"Total tokens: {total} > max_length: {max_length}")
    return total <= max_length