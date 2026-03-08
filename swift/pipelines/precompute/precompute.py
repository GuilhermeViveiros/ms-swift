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
        
        new_dataset = []
        for row in tqdm(dataset, desc='Precomputing'):
            try:
                encoded = self.template.encode(row)
                num_tokens = len(encoded['input_ids'])
                row['lengths'] = num_tokens
            except Exception as e:
                print("Error encoding row: ", e)
                continue
                # row["num_tokens"] = -1
            new_dataset.append(row)
            
        write_to_jsonl(output_path, new_dataset)    
        logger.info(f'Saved {len(dataset)} rows to {output_path}')
        return len(dataset)

    def run(self):
        args = self.args
        if not args.dataset and not args.val_dataset:
            raise ValueError('Please specify --dataset or --val_dataset.')
        print("Loading dataset..., dataset: ", args.dataset)
        dataset_kwargs = args.get_dataset_kwargs()
        train_dataset, val_dataset = None, None
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
        
        # extract name from output_dir
        #import pdb; pdb.set_trace()
        dataset_name = args.dataset[0].split('/')[-1]
        
        dataset_path = os.path.join(args.output_dir, dataset_name)
        #val_path = os.path.join(args.output_dir, dataset_name
        #import pdb; pdb.set_trace()
        logger.info(f'Dataset path: {dataset_path}')
        n_train = self._process_dataset(train_dataset, dataset_path)
        #n_val = self._process_dataset(val_dataset, val_path)
        logger.info(f'Precompute complete: train={n_train}')


def precompute_main(args: Optional[Union[List[str], PrecomputeArguments]] = None):
    return SwiftPrecompute(args).main()


    
# export ROOT_IMAGE_DIR=/e/scratch/jureap126/gviveiros/tvision/vision-data/images && swift precompute --model /e/scratch/jureap126/gviveiros/hf_models/TowerVision-2B --dataset_num_proc 8 --template llava_next_gemma2_hf --dataset /e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/pre_computed_jsonl/llava-next-cc-ocr-multi-lan-train.jsonl --output_dir /e/scratch/jureap126/gviveiros/tvision/vision-data/llava_datasets/filtered_data