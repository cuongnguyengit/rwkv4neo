from datasets import load_dataset
import os

path_to_file = '../data/train/'
MAX_PROC = 8


raw_datasets = load_dataset(
                    "text",
                    data_files=[os.path.join(path_to_file, i) for i in os.listdir(path_to_file) if
                                str(i).endswith(".txt")],
                    # cache_dir="/content/drive/MyDrive/llm/cache/",
                    cache_dir="../cache/",
                    # cache_dir="/kaggle/working/cache/",
                    # cache_dir=args.cache_dir,
                    num_proc=1,
                    # keep_in_memory=True,
                    sample_by="paragraph"
                )

raw_datasets['train'].to_json("mydataset.json", force_ascii=False)

print(raw_datasets)


# python tools/preprocess_data.py --input ./mydataset.jsonl --output-prefix ./data/mydataset --vocab ../checkpoint/vitok20k/tokenizer.json --dataset-impl mmap --tokenizer-type HFTokenizer --append-eod
# python tools/preprocess_data.py --input ./mydataset.jsonl --output-prefix ./data/mydataset2 --vocab ../checkpoint/vitok40k/tokenizer.json --dataset-impl mmap --tokenizer-type HFTokenizer --append-eod