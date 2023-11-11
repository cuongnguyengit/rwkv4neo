########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime
from itertools import chain
import os
import logging
from datasets import load_dataset
from random import randint

# MAX_PROC = os.cpu_count()
MAX_PROC = 8

logging.getLogger("datasets").setLevel(logging.ERROR)

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if args.data_type == "binidx":
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

            self.data = MMapIndexedDataset(args.data_file)
            self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
            rank_zero_info(f"Data has {self.data_size} tokens.")

            if args.my_qa_mask > 0:
                # self.data_pile = MMapIndexedDataset('/fsx/pile/pile_20B_tokenizer_text_document')
                self.data_pile = MMapIndexedDataset('/fsx/pile_deduped/pile_0.87_deduped_text_document')
                self.data_pile_size = len(self.data_pile._bin_buffer) // self.data._index._dtype_size
            else:
                self.data_pile = None
                self.data_pile_size = 0

            if args.my_pile_stage > 0:
                # assert self.data_size == 332115325534 and self.vocab_size == 50277
                self.samples_per_epoch = args.epoch_steps * args.real_bsz
                assert self.samples_per_epoch == 40320
                rank_zero_info(f"########## Pile 20b-tokenized stage {args.my_pile_stage} ##########")
                dataset_slot = self.data_size // args.ctx_len
                if args.my_pile_stage != 4:
                    assert MaybeIsPrime(args.magic_prime)
                    assert args.magic_prime % 3 == 2
                    assert args.magic_prime / dataset_slot > 0.99 and args.magic_prime / dataset_slot <= 1
        elif args.data_type == "numpy":
            self.data = np.load(args.data_file).astype("int")
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens.")
        elif args.data_type == "uint16":
            self.data = np.fromfile(args.data_file, dtype=np.uint16).astype("int32").reshape(-1, args.my_sample_len)
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            self.data_size = self.data.shape[0]
            rank_zero_info(f"Data has {self.data_size} samples.")
        elif args.data_type == "wds_img":
            self.vocab_size = -1
            self.data_size = -1
            self.data = None
            self.error_count = 0

        elif args.data_type == 'utf-8':
            rank_zero_info("load data...")
            path_to_file = args.data_file
            if os.path.isdir(path_to_file):
                raw_datasets = load_dataset(
                    "text",
                    data_files=[os.path.join(path_to_file, i) for i in os.listdir(path_to_file) if
                                str(i).endswith(".txt")],
                    cache_dir="/content/drive/MyDrive/llm/cache/",
                    # cache_dir="/workspace/cache/",
                    # cache_dir="/kaggle/working/cache/",
                    # cache_dir=args.cache_dir,
                    num_proc=MAX_PROC,
                    # keep_in_memory=True,
                    sample_by="paragraph"
                )
            elif os.path.isfile(path_to_file):
                raw_datasets = load_dataset(
                    "text",
                    data_files=path_to_file,
                    cache_dir="/content/drive/MyDrive/llm/cache/",
                    # cache_dir="/kaggle/working/cache/",
                    # cache_dir=args.cache_dir,
                    num_proc=MAX_PROC,
                    # keep_in_memory=True,
                    sample_by="paragraph"
                )
            else:
                raise "File error " + path_to_file

            # txt = open(path_to_file, "r", encoding=args.data_type).read()
            # from tokenization_phobert_fast import PhobertTokenizerFast
            os.environ["TOKENIZERS_PARALLELISM"] = "False"
            # tknz = PhobertTokenizerFast("./data/vocab.txt", "./data/bpe.codes", "./data/tokenizer.json")
            # self.vocab_size = 64256  # 251 * 256
            from transformers import AutoTokenizer
            # tknz = AutoTokenizer.from_pretrained("/content/drive/MyDrive/llm/checkpoint/rwkv4/")
            tknz = AutoTokenizer.from_pretrained("/content/drive/MyDrive/llm/checkpoint/vitok20k/")
            # tknz = AutoTokenizer.from_pretrained("/workspace/checkpoint/rwkv4c/")
            # tknz = AutoTokenizer.from_pretrained("/kaggle/input/vietnamesellmdatasets/")
            # tknz = AutoTokenizer.from_pretrained(args.tokenizer_path)

            # filter_datasets = raw_datasets.filter(
            #     lambda ex: len(ex['text']) > 0 and not ex['text'].isspace(),
            #     num_proc=MAX_PROC,
            #     desc=f'Filter empty datasets')

            def tokenize_function(examples):
                examples['text'] = [text.strip() + "<|endoftext|>" for text in examples['text']]
                output = tknz(examples['text'])
                return output

            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=MAX_PROC,
                remove_columns=['text'],
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
            )

            block_size = args.ctx_len + 1

            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
                # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
                total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                # print(len(result["input_ids"]), len(result["input_ids"][0]))
                # result["labels"] = result["input_ids"][:, 1:]
                # result["input_ids"] = result["input_ids"][:, :-1]
                return result

            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=12,
                load_from_cache_file=True,
                desc=f"Grouping texts in chunks of {block_size}",
            )

            self.vocab_size = tknz.vocab_size
            # self.data = tknz.encode(open(path_to_file, "r", encoding=args.data_type).read())
            # self.data_size = len(self.data)

            self.data = lm_datasets['train']
            self.data_size = len(self.data)

            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            rank_zero_info(f"Data has {self.data_size} samples.")
        elif args.data_type == 'jsonl':
            rank_zero_info("load data...")
            path_to_file = args.data_file
            if os.path.isdir(path_to_file):
                json_datasets = load_dataset(
                    "json",
                    data_files=[os.path.join(path_to_file, i) for i in os.listdir(path_to_file) if
                                str(i).endswith(".json") or str(i).endswith(".jsonl")],
                    # cache_dir="/content/drive/MyDrive/llm/cache/",
                    # cache_dir="/workspace/cache/",
                    num_proc=MAX_PROC,
                    # keep_in_memory=True,
                    cache_dir=args.cache_path,
                )
            elif os.path.isfile(path_to_file):
                json_datasets = load_dataset(
                    "json",
                    data_files=path_to_file,
                    # cache_dir="/content/drive/MyDrive/llm/cache/",
                    # cache_dir="/workspace/cache/",
                    num_proc=MAX_PROC,
                    # keep_in_memory=True,
                    cache_dir=args.cache_path,
                )
            else:
                raise "File error " + path_to_file
            os.environ["TOKENIZERS_PARALLELISM"] = "False"
            from transformers import AutoTokenizer
            # tknz = AutoTokenizer.from_pretrained("/content/drive/MyDrive/llm/checkpoint/vitok20k/")
            # tknz = AutoTokenizer.from_pretrained("/workspace/checkpoint/rwkv4c/")
            tknz = AutoTokenizer.from_pretrained(args.tokenizer_path)

            block_size = args.ctx_len + 1

            def tokenize_function(examples):
                return tknz(examples['text'])

            # json_datasets = json_datasets.filter(function=lambda sample: len(sample['text'].split()) < 2000)

            rank_zero_info(json_datasets)
            rank_zero_info(json_datasets['train'])

            tokenized_datasets = json_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=MAX_PROC,
                remove_columns=['text'],
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
            )

            tokenized_datasets = tokenized_datasets.filter(function=lambda sample: len(sample['input_ids']) <= block_size * 2 + 5)

            rank_zero_info(tokenized_datasets)

            def group_texts(examples):
                # examples['choice_mask'] = []
                result = {'input_ids': [], 'attention_mask': [], 'choice_mask': []}

                # choice_masks = []

                for i in range(len(examples['input_ids'])):
                    input_ids = examples['input_ids'][i]
                    attention_mask = examples['attention_mask'][i]

                    choice_mask = []

                    if args.qa_mask > 0:
                        result['input_ids'].append(input_ids)
                        result['attention_mask'].append(attention_mask)
                        is_qa = False
                        j = 0
                        while j + 3 < len(input_ids):
                        # for j in range(0, len(input_ids), 3):
                            k = j + 4
                            if input_ids[j: j + 3] == [15960, 27, 222]:
                                for k in range(j + 4, min(len(input_ids), j + 4 + block_size)):
                                    if input_ids[k] == 631:
                                        break

                                if input_ids[k + 1: k + 4] == [11827, 27, 222]:
                                    # choice_mask +=  [e for e in range(k + 4, min(len(input_ids), j + block_size))]
                                    for e in range(k + 4, min(len(input_ids), j + block_size)):
                                        choice_mask.append(e)
                                        # iid = input_ids[j: e]
                                        # atm = attention_mask[j: e]
                                        # n = len(iid)
                                        # if n < block_size:
                                        #     iid += [0] * (block_size - len(iid))
                                        #     atm += [0] * (block_size - len(atm))
                                        #
                                        # result['input_ids'] += [iid]
                                        # result['attention_mask'] += [atm]
                                    is_qa = True
                            elif input_ids[j: j + 3] == [11827, 27, 222]:
                                for k in range(j + 4, min(len(input_ids), j + 4 + block_size)):
                                    if input_ids[k] == 631:
                                        break
                                # choice_mask += [e for e in range(j + 4, k + 1)]
                                for e in range(j + 4, k + 1):
                                    choice_mask.append(e)
                                    # iid = input_ids[e - block_size: e]
                                    # atm = attention_mask[e - block_size: e]
                                    # n = len(iid)
                                    # if n < block_size:
                                    #     iid += [0] * (block_size - len(iid))
                                    #     atm += [0] * (block_size - len(atm))

                                    # result['input_ids'] += [iid]
                                    # result['attention_mask'] += [atm]
                                is_qa = True
                            j += 3

                        # if not is_qa:
                        #     iid = input_ids[: block_size]
                        #     atm = attention_mask[: block_size]
                        #
                        #     n = len(iid)
                        #     if n < block_size:
                        #         iid += [0] * (block_size - len(iid))
                        #         atm += [0] * (block_size - len(atm))
                        #
                        #         iid[n] = 631
                        #         atm[n] = 1
                        #     else:
                        #         iid[-1] = 631
                        #
                        #     result['input_ids'] += [iid]
                        #     result['attention_mask'] += [atm]
                        result['choice_mask'].append(choice_mask)
                    else:
                        total_length = (len(input_ids) // block_size + 1) * block_size
                        input_ids += [0] * (total_length - len(input_ids))
                        attention_mask += [0] * (total_length - len(attention_mask))

                        for j in range(0, total_length, block_size // 2):
                            iid = input_ids[j: j + block_size]
                            atm = attention_mask[j: j + block_size]

                            if len(iid) < block_size:
                                d = block_size - len(iid)
                                iid = input_ids[j - d: j + block_size - d]
                                atm = attention_mask[j - d: j + block_size - d]

                            result['input_ids'] += [iid]
                            result['attention_mask'] += [atm]

                return result

            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=MAX_PROC,
                load_from_cache_file=True,
                desc=f"Grouping texts in chunks of {block_size}",
            )

            self.vocab_size = tknz.vocab_size
            # self.data = tknz.encode(open(path_to_file, "r", encoding=args.data_type).read())
            # self.data_size = len(self.data)

            self.data = lm_datasets['train'].shuffle()
            self.data_size = len(self.data)

            print(self.data_size)

            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            rank_zero_info(f"Data has {self.data_size} samples.")
        else:
            if args.data_type == "dummy":
                rank_zero_info("Building dummy data...")
                self.data = ""
                for i in range(100000):
                    aa = (i) % 10000
                    bb = (i * i) % 10000
                    cc = aa + bb
                    self.data += f".{aa}+{bb}={cc}."
            else:
                self.data = open(args.data_file, "r", encoding=args.data_type).read()
            rank_zero_info("Building token list...")
            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)
            # rank_zero_info()
            # for u in unique:
            #     print(u, end=' ')
            # rank_zero_info('\n\n')
            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(f"{args.proj_dir}/vocab.json", "w", encoding="utf-16le") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}


    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz * self.args.agb

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        block_size = args.ctx_len + 1
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        if args.data_type == "wds_img":
            def init_wds(self, bias=0):
                def identity(x):
                    return x            
                import webdataset as wds
                import torchvision.transforms as transforms
                # img_transform = transforms.Compose(
                #     [transforms.CenterCrop(256)]
                # )
                img_transform = transforms.Compose([
                    transforms.CenterCrop(512),
                    transforms.Resize((args.my_img_size))
                ])
                self.data_raw = wds.WebDataset(args.data_file, resampled=True).shuffle(10000, initial=1000, rng=random.Random(epoch*100000+rank+bias*1e9)).decode("torchrgb").to_tuple("jpg", "json", "txt").map_tuple(img_transform, identity, identity)
                for pp in self.data_raw.pipeline:
                    if 'Resampled' in str(pp):
                        pp.deterministic = True
                        def worker_seed():
                            return rank*100000+epoch+bias*1e9
                        pp.worker_seed = worker_seed
                self.data = iter(self.data_raw)
                # print(f"WebDataset loaded for rank {rank} epoch {epoch}")
            if self.data == None:
                init_wds(self)
            trial = 0
            while trial < 10:
                try:
                    dd = next(self.data) # jpg, json, txt
                    break
                except:
                    print(f'[dataloader error - epoch {epoch} rank {rank} - trying a new shuffle]')
                    self.error_count += 1
                    init_wds(self, self.error_count)
                    trial += 1
                    pass
            # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} {dd[2]}")
            # with open(f"sample_{rank}.txt", "a", encoding="utf-8") as tmp:
            #     tmp.write(f"epoch {epoch} idx {idx} rank {rank}/{world_size} {int(dd[1]['key'])}\n")
            return dd[0], dd[2]
        else:
            if args.data_type in ["uint16", "utf8", "jsonl", 'utf-8']:
                # i = np.random.randint(0, self.data_size-1)
                s = idx * self.data_size / self.__len__()
                # e = (idx + 1) * self.data_size / self.__len__() - 1
                e = max((idx + 1) * self.data_size / self.__len__() - 1, s + 1)
                r = np.random.randint(s, e)

                if r >= self.data_size:
                    i = np.random.randint(0, self.data_size - 1)
                else:
                    i = r
                # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} from {s} to {e} -> choose {i}")
                dix = self.data[i]["input_ids"]

                if args.qa_mask > 0:
                    choice_mask = self.data[i]['choice_mask']
                    choice = choice_mask[randint(0, len(choice_mask) - 1)]
                    dix = dix[choice - block_size: choice]
                    dix = dix + [0] * (block_size - len(dix))

                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)
            else:
                ctx_len = args.ctx_len
                req_len = ctx_len + 1
                magic_prime = args.magic_prime
                data = self.data

                if args.my_pile_stage > 0:
                    ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

                    if args.my_qa_mask > 0:
                        ii_orig = ii
                        if ii % 2 == 0:
                            ii = -1
                            data = self.data_pile
                        else:
                            ii = ii // 2
                    if data == self.data_pile:
                        i = np.random.randint(0, self.data_pile_size - req_len)
                    else:
                        if args.my_pile_stage == 4 or ii < args.my_random_steps:
                            # cheat: pick a random spot in dataset
                            if args.my_pile_version == 1:
                                i = np.random.randint(0, self.data_size - req_len)
                            else:
                                i = np.random.randint(0, self.data_size)
                        else:
                            ii = ii - args.my_random_steps
                            factor = (math.sqrt(5) - 1) / 2
                            factor = int(magic_prime * factor)
                            i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
                            i = i + args.my_pile_shift
                    # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")
                else:
                    s = idx * self.data_size / self.__len__()
                    e = max((idx + 1) * self.data_size / self.__len__() - 1, s + 1)
                    i = min(np.random.randint(s, e), self.data_size - req_len)

                    # cheat: pick a random spot in dataset
                    # i = np.random.randint(0, self.data_size - req_len)

                if args.data_type == "binidx":
                    dix = data.get(idx=0, offset=i, length=req_len).astype(int)
                    # if args.my_pile_version == 1:
                    #     dix = data.get(idx=0, offset=i, length=req_len).astype(int)
                    # else:
                    #     # self.data : cutoff, chunk_count, data
                    #     for j in range(len(data)):
                    #         if i < data[j][0]:
                    #             ii = i
                    #             i = (i - (data[j-1][0] if j > 0 else 0)) % data[j][1]
                    #             dix = data[j][2].get(idx=0, offset=i, length=req_len).astype(int)
                    #             # print(ii, j, i)
                    #             break
                elif args.data_type == "numpy":
                    dix = data[i : i + req_len]
                else:
                    dix = [self.stoi[s] for s in data[i : i + req_len]]

                if args.my_qa_mask == 1:
                    if data == self.data_pile:
                        z = [1] * ctx_len
                    else:
                        z = [0] * ctx_len
                        z_sum = 0
                        isGood = False
                        for i in range(3, ctx_len):
                            if dix[i] == 27 and dix[i-1] == 34 and dix[i-2] == 187 and dix[i-3] == 187:
                                isGood = True
                            if dix[i] == 0:
                                isGood = False
                            if isGood:
                                z[i] = 1
                                z_sum += 1
                        if z_sum == 0:
                            z = [1] * ctx_len
                            i = np.random.randint(0, self.data_pile_size - req_len)
                            dix = self.data_pile.get(idx=0, offset=i, length=req_len).astype(int)
                    z = torch.tensor(z, dtype=torch.bfloat16)

                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)

                # if ii_orig < 50:
                #     # if rank == 1:
                #     print('rank', rank, 'i', ii_orig, ii, i, 'x', x[:5], '...', x[-5:])
                # else:
                #     exit(0)

                if args.my_qa_mask == 1:
                    return x, y, z

            return x, y
