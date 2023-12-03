# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


import ftfy
import json
from langdetect import detect
# import numpy as np
import time
# import os
import sys
# from pyvi.ViTokenizer import tokenize, spacy_tokenize

from tokenizer import tokenize, split_into_sentences

MIN_DOCUMENT_LENGHT = 50
AVG_CHAR_WORD = 5


def print_progress(prefix, start_time, num_docs, num_fixed_text,
                   num_non_english_docs, chars_non_english_docs,
                   num_small_docs, chars_small_docs, skipping_docs):

    string = prefix + ' | '
    string += 'elapsed time: {:.2f} | '.format(time.time() - start_time)
    string += 'documents: {} | '.format(num_docs)
    string += 'fixed text: {} | '.format(num_fixed_text)
    string += 'non-{}: {} | '.format(lang, num_non_english_docs)
    # string += 'non-english chars: {} | '.format(chars_non_english_docs)
    string += 'small docs: {} | '.format(num_small_docs)
    string += 'skipping docs: {} | '.format(skipping_docs)
    # string += 'small docs chars: {}'.format(chars_small_docs)
    print(string, flush=True)


def filter_corpus(filename, out_filename, print_interval=50000):

    print(' > filtering {}'.format(filename))

    # tokenizer = Tokenizer(cache_dir='./cache')

    num_docs = 0
    num_written_docs = 0
    num_small_docs = 0
    num_fixed_text = 0
    num_non_english_docs = 0
    chars_non_english_docs = 0
    chars_small_docs = 0
    skipping_docs = 0
    start_time = time.time()

    with open(out_filename, 'wb') as f:
        with open(filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                try:
                    num_docs += 1
                    line = line.strip()
                    try:
                        myjson = json.loads(line)
                    except:
                        myjson = {"text": line}
                    # Fix text
                    text = ftfy.fix_text(myjson['text'])
                    if text != myjson['text']:
                        num_fixed_text += 1
                    myjson['text'] = text
                    # Detect language.
                    # if detect(text) != 'en':
                    if detect(text) != lang:
                        # print(f'[non-{lang} text]', myjson)
                        num_non_english_docs += 1
                        chars_non_english_docs += len(text)
                        continue
                    # On average each token is 5 characters so 8 is an
                    # upper bound.
                    if len(text) < (AVG_CHAR_WORD * MIN_DOCUMENT_LENGHT):
                        # tokens = tokenizer.tokenize_document(text)
                        # tokens = [i for i in tokenize(text)]
                        tokens = [i for i in split_into_sentences(text)]
                        if len(tokens) < MIN_DOCUMENT_LENGHT:
                            # print('[small document, skipping]:', myjson)
                            num_small_docs += 1
                            chars_small_docs += len(text)
                            continue

                    # print(myjson)
                    myjson = json.dumps(myjson, ensure_ascii=False)
                    f.write(myjson.encode('utf-8'))
                    f.write('\n'.encode('utf-8'))
                    num_written_docs += 1
                    if num_docs % print_interval == 0:
                        print_progress('[PROGRESS]', start_time, num_docs,
                                       num_fixed_text, num_non_english_docs,
                                       chars_non_english_docs,
                                       num_small_docs, chars_small_docs, skipping_docs)
                except Exception as e:
                    # print('    skipping ', line, e)
                    skipping_docs += 1

    print_progress('[FINAL]', start_time, num_docs,
                   num_fixed_text, num_non_english_docs,
                   chars_non_english_docs,
                   num_small_docs, chars_small_docs, skipping_docs)


if __name__ == '__main__':

    print('building gpt2 dataset ...')

    lang = sys.argv[1]
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]

    print('lang is {}'.format(lang))
    print('will be reading {}'.format(input_filename))
    print('and will write the results to {}'.format(output_filename))

    filter_corpus(input_filename, output_filename)


