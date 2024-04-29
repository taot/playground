import os
import datasets
import torch
from torch import nn

import train
from config import config

ds = datasets.load_dataset("wmt/wmt19", "zh-en")
print(ds)

tokenizer = train.get_or_build_tokenizer(config, ds, "en")

print(tokenizer)
