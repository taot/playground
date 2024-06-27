from collections import defaultdict
from typing import Dict, List

import datasets
import pandas as pd
import plotly.express as px
import tqdm
from datasets import Dataset
from tokenizers import Tokenizer

import train
from config import get_config


def get_seq_lengths(ds: Dataset, tokenizer_map: Dict[str, Tokenizer], split: str) -> Dict[str, List[int]]:
    print(f"get_seq_lengths: split = {split}")
    seq_lengths_map = defaultdict(list)
    max_seq_len_map = defaultdict(lambda: 0)

    count = 0

    with tqdm.tqdm(total=len(ds)) as progress:
        for item in ds:
            for lang, tokenizer in tokenizer_map.items():
                text = item["translation"][lang]
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                seq_len = len(ids)
                seq_lengths_map[f"{lang}-{split}"].append(seq_len)
                max_seq_len_map[f"{lang}-{split}"] = max(max_seq_len_map[f"{lang}-{split}"], seq_len)

            progress.update(1)

            count += 1
            # if count > 1000:
            #     break

    print(dict(max_seq_len_map))

    return seq_lengths_map


def main() -> None:
    config = get_config()

    langs = ["en", "zh"]
    splits = ["train", "validation", "test"]

    tokenizer_map = {lang: train.get_tokenizer(config, lang) for lang in langs}

    for split in splits:
        ds = datasets.load_dataset(config["dataset"], name=config["dataset_config_name"], split=split)
        seq_length_map = get_seq_lengths(ds, tokenizer_map, split)
        for key, lst in seq_length_map.items():
            df = pd.DataFrame(lst, columns=[key])
            fig = px.histogram(df, x=key)
            fig.show()


if __name__ == "__main__":
    main()
