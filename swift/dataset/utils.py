# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import numpy as np
import os
import tempfile
from datasets import Dataset as HfDataset
from modelscope.hub.utils.utils import get_cache_dir
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from swift.template import Template
from swift.utils import get_logger
from .preprocessor import RowPreprocessor
from .token_estimate import (
    estimate_tokens_for_mllm_row,
    ensure_vision_tokens_in_row
)

logger = get_logger()

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

def filter_small_images(
    dataset: HfDataset,
    processor: Any,
    max_length: int,
    *,
    num_proc: int = 1,
    desc: str = 'filter_small_images',
) -> HfDataset:
    """
    Filter rows with small images.
    Load image with PIl and check size
    If size is less than 50x50, filter out
    Also if channel last filter out
    """
    def _filter_small_images(row: Dict[str, Any]) -> bool:
        """Filter rows with small images."""
        if not ensure_vision_tokens_in_row(row):
            return False
       
        image_sizes = _get_image_sizes_from_row(row)
        if not image_sizes:
            return True
        h, w = image_sizes[0][0], image_sizes[0][1]  # _get_image_size returns (height, width)
        if h * w < 50 * 50:
            return False
        return True
        
    return dataset.filter(_filter_small_images, num_proc=num_proc*4, desc=desc)

def filter_and_annotate_by_token_length(
    dataset: HfDataset,
    processor: Any,
    max_length: int,
    *,
    num_proc: int = 1,
    desc: str = 'map_and_filter_by_lengths',
    use_lightweight: bool = True,
) -> HfDataset:
    """
    Map (add lengths) and filter dataset rows that exceed max_length.
    Keeps rows where ensure_vision_tokens_in_row is True and total tokens <= max_length.
    Adds 'lengths' column to accepted samples for group_by_length.

    Args:
        dataset: HuggingFace Dataset.
        processor: Processor.
        max_length: Maximum sequence length.
        num_proc: Number of processes for dataset.map/filter.
        desc: Progress bar description.
        use_lightweight: If True, use fast processor-based estimation. If False,
            use template.encode() for exact token count (slower, matches training).

    Returns:
        Filtered dataset with 'lengths' column on accepted samples.
    """
    def _map_add_lengths(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Add lengths to each row; use -1 for rejected rows."""
        n = len(batch['messages']) if 'messages' in batch else len(next(iter(batch.values())))
        rows = [{k: batch[k][i] for k in batch} for i in range(n)]
        lengths = []
        for r in rows:
            if not ensure_vision_tokens_in_row(r):
                lengths.append(-1)
            else:
                v_tokens, total = estimate_tokens_for_mllm_row(r, processor=processor)
                if v_tokens > 0 and v_tokens < 100:
                    # ignore this row
                    lengths.append(-1)
                lengths.append(total if 0 <= total <= max_length else -1)
        return {**batch, 'lengths': lengths}

    def _keep_batch(batch: Dict[str, List[Any]]) -> List[bool]:
        """Keep rows where lengths >= 0 (accepted)."""
        return [l >= 0 for l in batch['lengths']]
    
    dataset = dataset.map(
        _map_add_lengths,
        batched=True,
        batch_size=256,
        num_proc=num_proc*6,
        desc=f'{desc}_map',
    )
    return dataset.filter(_keep_batch, batched=True, batch_size=256, num_proc=num_proc*6, desc=desc)



def sample_dataset(
        dataset: HfDataset,
        dataset_sample: Optional[int],
        shuffle: bool = True,
        random_state: Optional[np.random.RandomState] = None,
        shuffle_all: bool = False,  # For compatibility, this defaults to False.
) -> HfDataset:
    """Sample dataset by a dataset_sample number
    Args:
        dataset: The dataset instance, iterable dataset is not supported
        dataset_sample: The sample number
        shuffle: Whether to perform random sampling on non-streaming datasets
        random_state: The random state
    Returns:
        The sampled dataset
    """
    if dataset_sample is None:
        return dataset

    n_repeat_sample = dataset_sample // len(dataset)
    n_remain_sample = dataset_sample % len(dataset)
    if n_repeat_sample >= 1 and n_remain_sample >= 1:
        logger.warning(f'dataset_sample:{dataset_sample} is greater than len(dataset):{len(dataset)}, '
                       'repeated sampling will be performed.')
    idx = np.tile(range(len(dataset)), n_repeat_sample)
    if random_state is None:
        random_state = np.random.RandomState()
    if n_remain_sample >= 1:
        if shuffle:
            idx_remain = random_state.permutation(len(dataset))[:n_remain_sample]
        else:
            idx_remain = np.arange(n_remain_sample)
        idx = np.concatenate([idx, idx_remain])
    if n_repeat_sample >= 1 and shuffle and shuffle_all:
        random_state.shuffle(idx)
    dataset = dataset.select(idx)
    return dataset


class LazyLLMDataset(Dataset):
    """This class if used to lazy tokenize the dataset, and skips bad ones when training"""

    def __init__(self,
                 dataset: HfDataset,
                 encode_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                 *,
                 n_try_fetch: int = 10,
                 strict: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 traceback_limit: int = 10) -> None:
        self.dataset = dataset
        self.encode_func = encode_func

        n_try_fetch = 1 if strict else min(n_try_fetch, len(self.dataset))
        assert n_try_fetch >= 1
        self.strict = strict
        self.n_try_fetch = n_try_fetch

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

        self.traceback_limit = traceback_limit
        self._traceback_counter = 0
        self._idx = 0
        self._idx_list = self.random_state.permutation(len(self.dataset)).tolist()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(idx, str):
            return self.dataset[idx]
        for i in range(self.n_try_fetch):
            n_try = i
            if i == 0:
                i = idx
            else:
                i = self._idx_list[self._idx]
                self._idx = (self._idx + 1) % len(self.dataset)
            data = self.dataset[i]
            try:
                return self.encode_func(data, return_length=True)
            except Exception:
                if n_try == self.n_try_fetch - 1 or self.strict:
                    if self.strict:
                        logger.warning('To avoid errors, you can pass `strict=False`.')
                    raise
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback
                    logger.info(traceback.format_exc())
                    logger.warning('👆👆👆There are errors in the template.encode, '
                                   'and another piece of data will be randomly selected.')
                    self._traceback_counter += 1

        raise ValueError('Failed to retrieve the dataset. You can avoid this issue by increasing `max_length` or '
                         'modifying the `truncation_strategy`.')

    def __len__(self) -> int:
        return len(self.dataset)


class EncodePreprocessor(RowPreprocessor):

    def __init__(self, template: 'Template'):
        super().__init__()
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row, return_length=True)


class AddLengthPreprocessor(EncodePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        encoded = super().preprocess(row)
        row['lengths'] = encoded['lengths']
        return row


TEMP_DIR_POOL = {}


def get_temporary_cache_files_directory(prefix=None):
    if prefix is None:
        import datasets.config
        prefix = datasets.config.TEMP_CACHE_DIR_PREFIX
    if prefix in TEMP_DIR_POOL:
        TEMP_DIR = TEMP_DIR_POOL[prefix]
    else:
        tmp_dir = os.path.join(get_cache_dir(), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        kwargs = {}
        parameters = inspect.signature(tempfile.TemporaryDirectory.__init__).parameters
        if 'ignore_cleanup_errors' in parameters:
            kwargs['ignore_cleanup_errors'] = True
        TEMP_DIR = tempfile.TemporaryDirectory(prefix=prefix, dir=tmp_dir, **kwargs)
        logger.info(f'create tmp_dir: {TEMP_DIR.name}')
        TEMP_DIR_POOL[prefix] = TEMP_DIR

    return TEMP_DIR.name
