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


def main():
    config = get_config()
    lang = "zh"

    # Create tokenizer
    dataset = datasets.load_dataset(config["dataset"], config["dataset_config_name"], split="train")
    tokenizer = train.build_bbpe_tokenizer(config, dataset, lang)

    # tokenizer = train.get_tokenizer(config, lang)
    # print(tokenizer.get_vocab_size())

    # Test tokenizer
    tokenizer = train.get_tokenizer(config, lang)
    print(tokenizer.get_vocab_size())

    if lang == "zh":
        strs = [
            "财务科",
            "上帝在挑战你，他说你是笨蛋",
            "当然，雷曼兄弟公司的倒闭和柏林墙的倒塌没有任何关系。",
            "然而，和1989年一样，2008-2009年很可能也能被视为一个划时代的改变，其带来的发人深省的后果将在几十年后仍能让我们感受得到。",
            "1989年，自由民主战胜了由苏联集团具体化并推崇的社会主义意识形态。",
            "注意到据报首席部长发表声明，表示主张在东加勒比国家组织的政治联盟范围内实现独立，但自力更生比独立更具优先地位",
            "简言之，美国正在试图团结该地区所有担心中国以邻为壑的贸易和汇率政策的国家。对美国来说，其他八个TPP国家总人口可达2亿，能够构成其仅次于中国、欧盟和日本的第四大出口市场。如果日本加入的话，TPP的重要性还能显著增加。",
        ]
    else:
        strs = [
            "Desertification monitoring and the impacts of climate change",
            "ILOAT, however, appeared to have the edge over UNAT.",
            "Even lke wanted to pose for a photo in front of that huge trophy.",
        ]

    for s in strs:
        encoding = tokenizer.encode(s, add_special_tokens=True)
        print(s)
        utils.print_encoding(encoding)

        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        print(f"decoded: {decoded}")

        print()


if __name__ == "__main__":
    main()
