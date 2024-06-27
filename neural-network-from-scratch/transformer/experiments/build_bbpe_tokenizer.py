from pathlib import Path
from typing import Dict, Any

import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers, processors, decoders
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

import train
import utils
from config import get_config


def build_tokenizer(config: Dict[str, Any], dataset: datasets.Dataset, lang: str):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2, show_progress=False)
    trainer = BpeTrainer(
        vocab_size=20000,
        show_progress=False,
        special_tokens=["<UNK>", "\n"],
        min_frequency=1
    )

    tokenizer.train_from_iterator(iterator=train.get_all_sentences(dataset, lang, limit=None, verbose=False), trainer=trainer)

    # tokenizer.post_processor = TemplateProcessing(
    #     single="[SOS] $0 [EOS]",
    #     pair="[SOS] $A [EOS] $B:1 [EOS]:1",
    #     special_tokens=[
    #         ("[SOS]", tokenizer.token_to_id("[SOS]")),
    #         ("[EOS]", tokenizer.token_to_id("[EOS]"))
    #     ],
    # )

    # tokenizer.post_processor = processors.ByteLevel()
    # tokenizer.decoder = decoders.ByteLevel()

    tokenizer.save(str(tokenizer_path))


def hg_cut(tokenizer, text):
    encoding = tokenizer.encode(text, add_special_tokens=False)

    tokens = tokenizer.encode(text).tokens
    pos = 0
    for i in range(len(tokens)):
        if tokens[i] == "[UNK]":
            tokens[i] = text[pos]
            pos += 1
        else:
            pos += len(tokens[i])
    return tokens


def main():
    config = get_config()
    lang = "zh"

    # Create tokenizer
    # dataset = datasets.load_dataset(config["dataset"], config["dataset_config_name"], split="train")
    # tokenizer = train.build_tokenizer(config, dataset, lang)

    # tokenizer = train.get_tokenizer(config, lang)
    # print(tokenizer.get_vocab_size())

    # Test tokenizer
    tokenizer = train.get_tokenizer(config, lang)
    print(tokenizer.get_vocab_size())
    # strs = [
    #     "Desertification monitoring and the impacts of climate change",
    #     "ILOAT, however, appeared to have the edge over UNAT.",
    #     "Even lke wanted to pose for a photo in front of that huge trophy.",
    # ]
    strs = [
        "上帝在挑战你，他说你是笨蛋",
        "当然，雷曼兄弟公司的倒闭和柏林墙的倒塌没有任何关系。",
        "然而，和1989年一样，2008-2009年很可能也能被视为一个划时代的改变，其带来的发人深省的后果将在几十年后仍能让我们感受得到。",
        "1989年，自由民主战胜了由苏联集团具体化并推崇的社会主义意识形态。",
    ]
    for s in strs:
        encoding = tokenizer.encode(s, add_special_tokens=True)
        print(s)
        utils.print_encoding(encoding)

        # decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        # print(f"decoded: {decoded}")

        output = hg_cut(tokenizer, s)
        print("|".join(output))

        print()


if __name__ == "__main__":
    main()
