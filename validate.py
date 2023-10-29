from transformers import (
    RwkvForCausalLM,
    RwkvConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

from datasets import load_dataset
import torch
import numpy as np
import re
from argparse import ArgumentParser
from typing import Any, Dict
import math

def remove_url_from_text(text: str):
    """Remove square brackets around linked text and (_URL_0_) after"""
    return re.sub(r"\[|\]|\(_URL_\d+_\)", "", text)

def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
    """Concatenate and tokenize the answers in flattened ELI5 data"""
    concatenated = [remove_url_from_text(" ".join(x)) for x in examples["text"]]
    return tokenizer(concatenated)


def chunk(examples: Dict[str, Any], chunk_size: int = 256) -> Dict[str, Any]:
    """Concatenate and chunk batches of data"""
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    return {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated.items()
    }


def set_labels(examples: Dict[str, Any]) -> Dict[str, Any]:
    """Add a labels column to the dataset which is a copy of input_ids"""
    examples["labels"] = examples["input_ids"].copy()
    return examples


if __name__ == "__main__":
    MODEL_NAME = "../checkpoint/rwkv4_vitok20k_l12_768_128/"
    DATASET = "../data/val/vnexpress.txt"
    CHUNK_SIZE = 128
    # TEST_SPLIT_SIZE = 0.2
    BATCH_SIZE = 128
    # DATASET_SPLIT = "train_asks[:500]"

    model = RwkvForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer.pad_token = tokenizer.unk_token

    dataset = load_dataset('text', data_files=DATASET, cache_dir="../cache/", sample_by="paragraph")
    # dataset = load_dataset(DATASET, split=DATASET_SPLIT)
    # dataset = dataset.train_test_split(test_size=TEST_SPLIT_SIZE)
    dataset = dataset.flatten()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Encode
    encoded_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=12,
        remove_columns=dataset["train"].column_names,
        desc="Tokenize",
        load_from_cache_file=True
    )

    # Chunk
    chunked_dataset = encoded_dataset.map(
        chunk,
        fn_kwargs={"chunk_size": CHUNK_SIZE},
        batched=True,
        num_proc=2,
        desc="Chunk",
        load_from_cache_file=True
    )

    # Label
    lm_dataset = chunked_dataset.map(
        set_labels,
        batched=True,
        num_proc=2,
        desc="Label",
        load_from_cache_file=True
    )

    training_args = TrainingArguments(
        output_dir=MODEL_NAME + "-" + DATASET,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        prediction_loss_only=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=len(lm_dataset["train"]) // BATCH_SIZE,
        bf16_full_eval=True,
        bf16=True,
        do_eval=True,
        per_device_eval_batch_size=BATCH_SIZE
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["train"],
        data_collator=data_collator,
    )

    # Evaluate before train
    eval_0 = trainer.evaluate()
    perplexity_0 = math.exp(eval_0["eval_loss"])

    print(eval_0)
    print(perplexity_0)

    # Train
    # trainer.train()

    # Evaluate after train
    # eval_f = trainer.evaluate()
    # perplexity_f = math.exp(eval_f["eval_loss"])