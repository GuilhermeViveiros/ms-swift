# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import torch
from datasets import Dataset as HfDataset
from tqdm import tqdm
from typing import List, Optional, Union

from swift.arguments import PrecomputeArguments
from swift.dataset import load_dataset
from swift.utils import check_json_format, get_logger, write_to_jsonl
from ..train import SwiftSft
from swift.dataset.token_estimate import ensure_vision_tokens_in_row, _get_image_sizes_from_row
logger = get_logger()

def filter_small_images(row: dict) -> bool:
    if not ensure_vision_tokens_in_row(row):
        return False

    image_sizes = _get_image_sizes_from_row(row)
    if not image_sizes:
        return True
    h, w = image_sizes[0][0], image_sizes[0][1]  # _get_image_size returns (height, width)
    if h * w < 50 * 50:
        return False
    return True
            

class SwiftPrecompute(SwiftSft):
    args_class = PrecomputeArguments
    args: PrecomputeArguments

    def __init__(self, args: Optional[Union[List[str], PrecomputeArguments]] = None) -> None:
        super(SwiftSft, self).__init__(args)
        self.train_msg = {}
        template_cls = self.args.template_meta.template_cls
        if template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self._prepare_model_tokenizer(**kwargs)
        self._prepare_template()

    def _process_dataset(self, dataset, output_path: str) -> int:
        """Process dataset rows, filter small images, write to JSONL. Uses parallel filter+map for speed."""
        if dataset is None or len(dataset) == 0:
            return 0

        num_proc = max(1, getattr(self.args, 'dataset_num_proc', 1) or 1)

        if isinstance(dataset, HfDataset):
            # Fast path: parallel filter (image loading) + parallel map (JSON formatting)
            start_size = len(dataset)
            filtered = dataset.filter(
                filter_small_images,
                num_proc=num_proc * 4,
                desc='Filtering small images',
            )
            end_size = len(filtered)
            rows = filtered.to_list()
        else:
            # Fallback for IterableDataset or other types
            start_size = len(dataset)
            rows = []
            for i in tqdm(range(len(dataset)), desc='Precomputing'):
                row = dataset[i]
                if isinstance(row, dict):
                    row = dict(row)
                else:
                    row = {k: row[k] for k in dataset.column_names}
                if not filter_small_images(row):
                    continue
                rows.append(check_json_format(row))
            end_size = len(rows)

        num_filtered = start_size - end_size
        percent_filtered = (100.0 * num_filtered / start_size) if start_size > 0 else 0.0
        logger.info(
            f"\n=== Dataset Filtering Report Precompute ===\n"
            f"\n    Initial size        : {start_size:,}"
            f"\n    Filtered out        : {num_filtered:,}"
            f"\n    Percentage filtered : {percent_filtered:.2f}%"
            f"\n    Remaining size      : {end_size:,}"
        )
        write_to_jsonl(output_path, rows)
        logger.info(f'Saved {len(rows)} rows to {output_path}')
        return len(rows)

    def run(self):
        args = self.args
        if not args.dataset and not args.val_dataset:
            raise ValueError('Please specify --dataset or --val_dataset.')
        print("Loading dataset..., dataset: ", args.dataset)
        dataset_kwargs = args.get_dataset_kwargs()
        train_dataset, val_dataset = None, None
         # extract name from output_dir
        dataset_name = args.output_dir.split('/')[-1]
        # remove it from output_dir
        output_dir = args.output_dir.replace(dataset_name, '')
        dataset_name = dataset_name + '.jsonl'
        print("Output directory: ", output_dir)
        print("Dataset name: ", dataset_name)

        # if dataset name already existits here: /e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/pre_computed_jsonl ignore
        if os.path.exists(os.path.join('/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/pre_computed_jsonl', dataset_name)):
            logger.info(f'Dataset {dataset_name} already exists in /e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/pre_computed_jsonl')
            return


        if args.dataset:
            train_dataset, val_dataset = load_dataset(
                args.dataset,
                split_dataset_ratio=args.split_dataset_ratio,
                shuffle=args.dataset_shuffle,
                template=self.template,
                **dataset_kwargs)
        if args.val_dataset:
            dataset_kwargs.pop('interleave_prob', None)
            _, val_dataset = load_dataset(
                args.val_dataset,
                split_dataset_ratio=1.0,
                shuffle=args.val_dataset_shuffle,
                template=self.template,
                **dataset_kwargs)
            if args.dataset:
                assert args.split_dataset_ratio == 0.
        
       
        
        dataset_path = os.path.join('/e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/pre_computed_jsonl', dataset_name)
        #val_path = os.path.join(args.output_dir, dataset_name)
        n_train = self._process_dataset(train_dataset, dataset_path)
        #n_val = self._process_dataset(val_dataset, val_path)
        logger.info(f'Precompute complete: train={n_train}')


def precompute_main(args: Optional[Union[List[str], PrecomputeArguments]] = None):
    return SwiftPrecompute(args).main()
