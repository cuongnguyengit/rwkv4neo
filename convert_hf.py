from transformers.models.rwkv.convert_rwkv_checkpoint_to_hf import PreTrainedTokenizerFast, AutoTokenizer, RwkvConfig, AutoModelForCausalLM, convert_state_dict, shard_checkpoint, WEIGHTS_INDEX_NAME
import torch
import os
import json
import gc

tokenizer = AutoTokenizer.from_pretrained("../checkpoint/vitok20k/")

tokenizer_file = "../checkpoint/vitok20k/tokenizer.json"
checkpoint_file = '../checkpoint/rwkv4_vitok20k_l12_768_128/rwkv-4.pth'
output_dir = '../checkpoint/rwkv4_vitok20k_l12_768_128/'


n_layer = 12
n_embd = 768
# args.ctx_len = 1024
# args.ctx_len = 4096
ctx_len = 128


# 1. If possible, build the tokenizer.
if tokenizer_file is None:
    print("No `--tokenizer_file` provided, we will use the default tokenizer.")
    vocab_size = 50277
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
else:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    vocab_size = len(tokenizer)
tokenizer.save_pretrained(output_dir)

config = RwkvConfig(
    vocab_size=vocab_size,
    num_hidden_layers=n_layer,
    hidden_size=n_embd,
)
config.save_pretrained(output_dir)

# 3. Download model file then convert state_dict

state_dict = torch.load(checkpoint_file, map_location="cpu")
state_dict = convert_state_dict(state_dict)

# 4. Split in shards and save
shards, index = shard_checkpoint(state_dict)
for shard_file, shard in shards.items():
    torch.save(shard, os.path.join(output_dir, shard_file))

if index is not None:
    save_index_file = os.path.join(output_dir, WEIGHTS_INDEX_NAME)
    # Save the index as well
    with open(save_index_file, "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)

    # 5. Clean up shards (for some reason the file PyTorch saves take the same space as the whole state_dict
    print(
        "Cleaning up shards. This may error with an OOM error, it this is the case don't worry you still have converted the model."
    )
    shard_files = list(shards.keys())

    del state_dict
    del shards
    gc.collect()

    for shard_file in shard_files:
        state_dict = torch.load(os.path.join(output_dir, shard_file))
        torch.save({k: v.cpu().clone() for k, v in state_dict.items()}, os.path.join(output_dir, shard_file))

del state_dict
gc.collect()





