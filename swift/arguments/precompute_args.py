# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from swift.utils import get_logger, to_abspath

from .base_args import BaseArguments

logger = get_logger()


@dataclass
class PrecomputeArguments(BaseArguments):
    """Arguments for precomputing token counts per dataset row.

    Loads datasets, computes exact token count per row using template.encode,
    and saves original data + num_tokens to JSONL.

    Args:
        output_dir (Optional[str]): Directory to save train.jsonl and val.jsonl.
            Defaults to ./precomputed_{timestamp}.
    """
    output_dir: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self._init_output_dir()

    def _init_output_dir(self):
        if self.output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = f'./precomputed_{timestamp}'
        self.output_dir = to_abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f'args.output_dir: `{self.output_dir}`')
