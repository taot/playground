import os
import datasets
import torch
from torch import nn

import train
from config import config

dataset_dict = datasets.load_dataset("wmt/wmt19", "zh-en", split="train")
print(dataset_dict)

tokenizer = train.get_or_build_tokenizer(config, dataset_dict, "en")

print(tokenizer)

sentence = dataset_dict["train"][2]["translation"]["en"]
print(sentence)

encoding = tokenizer.encode(sentence, add_special_tokens=True)
print(encoding.tokens)
print(encoding.ids)

print(tokenizer.decode(encoding.ids, skip_special_tokens=False))
