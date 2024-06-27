import os
from typing import Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import train
from config import get_config

SEQ_LEN_LIMIT = 100

NUM_ROWS_LIMIT = 100000

SOURCE_DIR = "/home/taot/data/ml_data/my_projects/experiments/opus-100/en-zh/"
TARGET_DIR = "/home/taot/data/huggingface/opus-100-short/en-zh/"

config = get_config()

total_count = 0

tokenizers = {
    "en": train.get_tokenizer(config, "en"),
    "zh": train.get_tokenizer(config, "zh"),
}


def is_too_long(translation: Dict[str, Any]) -> bool:
    for lang in ["en", "zh"]:
        text = translation[lang]
        ids = tokenizers[lang].encode(text, add_special_tokens=False).ids
        if len(ids) > SEQ_LEN_LIMIT:
            return True

    return False


def process_file(src_file: str, tgt_file: str) -> None:
    table = pq.read_table(src_file)
    df = table.to_pandas()

    data = []

    with tqdm(total=len(df)) as progress:
        for index, row in df.iterrows():
            progress.update(1)
            translation = row["translation"]
            if is_too_long(translation):
                continue

            data.append(translation)
            global total_count
            total_count += 1

            # if total_count > NUM_ROWS_LIMIT:
            #     break

    tgt_df = pd.DataFrame({"translation": data})
    tgt_table = pa.Table.from_pandas(tgt_df, preserve_index=False)
    pq.write_table(tgt_table, tgt_file)


def main():
    files = os.listdir(SOURCE_DIR)
    print(files)

    for file in files:
        src_file = SOURCE_DIR + file
        tgt_file = TARGET_DIR + file
        print(f"Processing {src_file}")
        process_file(src_file, tgt_file)

        if total_count > NUM_ROWS_LIMIT:
            break


if __name__ == "__main__":
    main()
