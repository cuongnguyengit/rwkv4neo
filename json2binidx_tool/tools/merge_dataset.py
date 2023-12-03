
import argparse
import multiprocessing
import os
import sys

import lm_dataformat as lmd
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import time
import tqdm
import ftfy

from tokenizer import build_tokenizer
import indexed_dataset
from threading import Semaphore


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        # required=True,
        default='../data/',
        help="Folder path contains jsonl files"
        "list",
    )
    group.add_argument("--ftfy", action="store_true", help="Use ftfy to clean text")
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        choices=["lazy", "cached", "mmap"],
        help="Dataset implementation to use. Default: mmap",
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        # required=True,
        default="HFTokenizer",
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "TiktokenTokenizer",
            "RWKVTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str,
        # default=None,
        default="../20B_tokenizer.json",
        help="Path to the vocab file"
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        # required=True,
        default='../data/text',
        help="Path to binary output file without suffix",
    )
    args = parser.parse_args()
    args.keep_empty = False
    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args



def main():
    args = get_args()
    tokenizer = build_tokenizer(args)
    all_bin_files = [os.path.join(args.input, file).replace(".bin", "") for file in os.listdir(args.input) if file.endswith(".bin")]

    des_builder = indexed_dataset.make_builder(
        "{}_{}_{}.bin".format(
            args.output_prefix, "text", "document"
        ),
        impl=args.dataset_impl,
        vocab_size=tokenizer.vocab_size,
    )
    pbar = tqdm.tqdm()
    for i, bin_file in enumerate(all_bin_files):
        if indexed_dataset.check_exist_dataset(bin_file, args.dataset_impl):
            des_builder.merge_file_(bin_file)
            pbar.set_description(
                f"Processed {i} - {bin_file} => size {len(des_builder)}"
            )
            if i != 0:
                pbar.update(1)

    des_builder.finalize("{}_{}_{}.idx".format(
            args.output_prefix, "text", "document"
        ))









if __name__ == '__main__':
    main()