import datasets
from datasets import Dataset
from tokenizers import Tokenizer
import tqdm

import train
from config import get_config


def get_max_seq_len_for_dataset(ds: Dataset, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, lang_src: str, lang_tgt: str) -> int:
    max_len_src = 0
    max_len_tgt = 0
    max_text_src = ""
    max_text_tgt = ""

    with tqdm.tqdm(total=len(ds)) as progress:
        for item in ds:
            text = item["translation"][lang_src]
            ids = tokenizer_src.encode(text).ids
            if len(ids) > max_len_src:
                max_len_src = len(ids)
                max_text_src = text

            text = item["translation"][lang_tgt]
            ids = tokenizer_tgt.encode(text, add_special_tokens=False).ids
            if len(ids) > max_len_tgt:
                max_len_tgt = len(ids)
                max_text_tgt = text

            progress.update(1)

    print(f"(lang = {lang_src}) max_len: {max_len_src}, max_text: {max_text_src}")
    print(f"(lang = {lang_tgt}) max_len: {max_len_tgt}, max_text: {max_text_tgt}")

    return max(max_len_src, max_len_tgt)


def main() -> None:
    config = get_config()

    tokenizer_src = train.get_tokenizer(config, "en")
    tokenizer_tgt = train.get_tokenizer(config, "zh")

    train_ds = datasets.load_dataset(config["dataset"], name=config["dataset_config_name"], split="train")
    val_ds = datasets.load_dataset(config["dataset"], name=config["dataset_config_name"], split="validation")

    max_len_train = get_max_seq_len_for_dataset(train_ds, tokenizer_src, tokenizer_tgt, "en", "zh")
    max_len_val = get_max_seq_len_for_dataset(val_ds, tokenizer_src, tokenizer_tgt, "en", "zh")

    print(f"max_seq_len (overall): {max(max_len_train, max_len_val)}")


if __name__ == "__main__":
    main()
