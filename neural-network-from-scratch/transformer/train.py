from typing import Dict, Any

import torch
from torch import nn

import datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_training_sentences(ds: datasets.Dataset, lang: str):
    print(f"get_all_training_sentences: lang = {lang}")
    count = 0
    for item in ds["train"]:
        if count > 10:
            return
        else:
            count += 1
            print(f"get_all_training_sentences: item = {item}")
            sentence = item["translation"][lang]
            print(f"get_all_training_sentences: {sentence}")
            yield sentence


def get_or_build_tokenizer(config: Dict[str, Any], ds: datasets.Dataset, lang: str) -> Tokenizer:
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    print(f"get_or_build_tokenizer: tokenizer_path = {tokenizer_path}")
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(iterator=get_all_training_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

    return tokenizer

