import json, time, random, os
import numpy as np
import torch
from torch.nn import functional as F
from rwkv_tokenizer import TRIE_TOKENIZER
from tokenizers import Tokenizer
import tiktoken
from transformers import PreTrainedTokenizerFast


time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

class TOKENIZER():
    def __init__(self, WORD_NAME):

        if WORD_NAME == "cl100k_base":
            self.tokenizer = tiktoken.get_encoding(WORD_NAME)
        elif WORD_NAME == "rwkv_vocab_v20230424.txt":
            self.tokenizer = TRIE_TOKENIZER(WORD_NAME)
        else:
            if WORD_NAME.endswith(".txt"):
                self.tokenizer = TRIE_TOKENIZER(WORD_NAME)
            elif WORD_NAME.endswith(".json"):
                self.tokenizer = Tokenizer.from_file(WORD_NAME)
            else:
                self.tokenizer = Tokenizer.from_pretrained(WORD_NAME)
        self.charMode = False

    def refine_context(self, context):
        context = context.strip().split("\n")
        for c in range(len(context)):
            context[c] = context[c].strip().strip("\u3000").strip("\r")
        context = list(filter(lambda c: c != "", context))
        context = "\n" + ("\n".join(context)).strip()
        if context == "":
            context = "\n"
        return context

    def encode(self, x):
        if "Tokenizer" in str(type(self.tokenizer)):
            return self.tokenizer.encode(x).ids
        else:
            return self.tokenizer.encode(x)

    def decode(self, x):
        return self.tokenizer.decode(x)

    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        if probs.device == torch.device("cpu"):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)

def MaybeIsPrime(number):
    if FermatPrimalityTest(number) and MillerRabinPrimalityTest(number):
        return True
    else:
        return False


def FermatPrimalityTest(number):
    if number > 1:
        for time in range(3):
            randomNumber = random.randint(2, number) - 1
            if pow(randomNumber, number - 1, number) != 1:
                return False
        return True
    else:
        return False


def MillerRabinPrimalityTest(number):
    if number == 2:
        return True
    elif number == 1 or number % 2 == 0:
        return False
    oddPartOfNumber = number - 1
    timesTwoDividNumber = 0
    while oddPartOfNumber % 2 == 0:
        oddPartOfNumber = oddPartOfNumber // 2
        timesTwoDividNumber = timesTwoDividNumber + 1

    for time in range(3):
        while True:
            randomNumber = random.randint(2, number) - 1
            if randomNumber != 0 and randomNumber != 1:
                break

        randomNumberWithPower = pow(randomNumber, oddPartOfNumber, number)

        if (randomNumberWithPower != 1) and (randomNumberWithPower != number - 1):
            iterationNumber = 1

            while (iterationNumber <= timesTwoDividNumber - 1) and (randomNumberWithPower != number - 1):
                randomNumberWithPower = pow(randomNumberWithPower, 2, number)
                iterationNumber = iterationNumber + 1
            if randomNumberWithPower != (number - 1):
                return False

    return True
