import os
import datasets
import torch
from torch import nn

import train
from config import config

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

dataset_dict = datasets.load_dataset("wmt/wmt19", "zh-en")
print(dataset_dict)


def train_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train_from_iterator(iterator=train.get_all_sentences(dataset_dict, "train", "en", verbose=False), trainer=trainer)
    tokenizer.save("exp_tokenizer.json")
    return tokenizer


def run_decode() -> None:
    tokenizer = Tokenizer.from_file("exp_tokenizer.json")
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    for sentence in train.get_all_sentences(dataset_dict, "validation", "en", limit=10, verbose=True):
        encoding = tokenizer.encode(sentence, sentence, add_special_tokens=True)
        print(encoding)

        print("ids: ", encoding.ids)
        print("type_ids: ", encoding.type_ids)
        print("tokens: ", encoding.tokens)
        print("offsets: ", encoding.offsets)
        print("special_tokens_mask: ", encoding.special_tokens_mask)

        decoded = tokenizer.decode(encoding.ids)
        print("decoded: ", decoded)

        print()


# train_tokenizer()

run_decode()
