import datasets

import train
from config import get_config


def main():
    config = get_config()
    lang = "en"
    dataset = datasets.load_dataset(config["dataset"], config["dataset_config_name"], split="train")
    train.build_tokenizer(config, dataset, lang)

    tokenizer = train.get_tokenizer(config, lang)
    print(tokenizer.get_vocab_size())


if __name__ == "__main__":
    main()
