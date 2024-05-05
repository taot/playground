from typing import Dict, Any, Optional

import torch
from tokenizers.processors import TemplateProcessing
from torch import nn

import datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

import tqdm


def get_all_sentences(dataset_dict: datasets.DatasetDict, key: str, lang: str, *, limit: Optional[int] = None, verbose: bool = False):

    def print_log(s: str) -> None:
        print(f"get_all_training_sentences: {s}")

    assert key in dataset_dict.keys(), f"dataset key must be one of {dataset_dict.keys()}"
    print_log(f"key = {key}, lang = {lang}, limit = {limit}")

    ds = dataset_dict[key]

    if limit is not None:
        assert limit >= 0, "limit must be greater than or equal to 0"
        ds = ds.select(range(limit))

    with tqdm.tqdm(total=len(ds)) as progress:
        for item in ds:
            sentence = item["translation"][lang]
            if verbose:
                print_log(sentence)
            progress.update(1)
            yield sentence


def get_or_build_tokenizer(config: Dict[str, Any], dataset_dict: datasets.DatasetDict, lang: str) -> Tokenizer:
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    print(f"get_or_build_tokenizer: tokenizer_path = {tokenizer_path}")
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        tokenizer.train_from_iterator(iterator=get_all_sentences(dataset_dict, "train", lang, limit=1000, verbose=False), trainer=trainer)

        tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $0 [EOS]",
            pair="[SOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", tokenizer.token_to_id("[SOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]"))
            ],
        )

        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer
