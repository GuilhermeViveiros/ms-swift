# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import numpy as np
import os
import tempfile
from datasets import Dataset as HfDataset
from modelscope.hub.utils.utils import get_cache_dir
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, Optional, Union

from swift.template import Template
from swift.utils import get_logger
from .preprocessor import RowPreprocessor
from .token_estimate import (
    estimate_tokens_for_mllm_row,
    ensure_vision_tokens_in_row
)

logger = get_logger()


def filter_dataset_by_length(
    dataset: HfDataset,
    template: Template,
    max_length: int,
    *,
    num_proc: int = 1,
    desc: str = 'filter_by_length',
    model_dir: Optional[str] = None,
) -> HfDataset:
    """
    Filter dataset rows that exceed max_length (estimated via template.encode).

    Useful for GKD/multimodal training to avoid collective shape mismatches from
    oversized sequences.

    Args:
        dataset: HuggingFace Dataset.
        template: Initialized template (processor inited).
        max_length: Maximum sequence length.
        num_proc: Number of processes for dataset.map.
        desc: Progress bar description.
        model_dir: Model path for TowerVision (e.g. utter-project/TowerVision-2B).
            Used for lightweight image-token estimation when filtering.

    Returns:
        Filtered dataset.
    """
    from typing import List
    def _keep_batch(batch: Dict[str, List[Any]]) -> List[bool]:
        """Process a batch of rows; return list of bools."""
        n = len(batch['messages']) if 'messages' in batch else len(next(iter(batch.values())))
        rows = [{k: batch[k][i] for k in batch} for i in range(n)]
        return [ensure_vision_tokens_in_row(r) and estimate_tokens_for_mllm_row(r, template, max_length, model_dir=model_dir) for r in rows]

    return dataset.filter(lambda row: _keep_batch(row), batched=True, batch_size=456, num_proc=num_proc*10, desc=desc)


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
