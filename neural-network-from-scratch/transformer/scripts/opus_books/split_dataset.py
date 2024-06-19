import os
from typing import Dict, Any, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import train
from config import get_config

SEQ_LEN_LIMIT = 50

NUM_ROWS_LIMIT = 100000

SOURCE_DIR = "/home/taot/data/ml_data/my_projects/experiments/wmt19-zh-en/"
TARGET_DIR = "/home/taot/data/huggingface/wmt19-short/zh-en-50-small/"


def write_data_to_parquet(data: List[Dict[str, Any]], filepath: str) -> None:
    df = pd.DataFrame({"translation": data})
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, filepath)


def process_file(src_file: str, tgt_dir: str) -> None:
    table = pq.read_table(src_file)
    df = table.to_pandas()

    ds_size = len(df)
    train_ds_size = int(0.9 * ds_size)
    val_ds_size = ds_size - train_ds_size

    train_data = []
    val_data = []

    count = 0

    with tqdm(total=len(df)) as progress:
        for index, row in df.iterrows():
            progress.update(1)
            translation = row["translation"]

            if count < train_ds_size:
                train_data.append(translation)
            else:
                val_data.append(translation)

            count += 1

    write_data_to_parquet(train_data, tgt_dir + "/train-00000-of-00001.parquet")
    write_data_to_parquet(val_data, tgt_dir + "/validation-00000-of-00001.parquet")


def main():
    src_file = "/home/taot/data/ml_data/my_projects/experiments/opus_books/en-it/train-00000-of-00001.parquet"
    tgt_dir = "/home/taot/data/huggingface/opus_books_split/en-it/"
    process_file(src_file, tgt_dir)


if __name__ == "__main__":
    main()
