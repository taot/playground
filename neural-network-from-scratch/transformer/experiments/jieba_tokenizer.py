from pathlib import Path

import datasets

import train
import utils

from jieba_tokenizer_py import JiebaTokenizer

from config import get_config

dataset_dict = datasets.load_dataset("wmt/wmt19", "zh-en", split="train")
print(dataset_dict)

config = get_config()
# tokenizer_path = Path(config["tokenizer_file"].format("zh"))
tokenizer_path = Path("/home/taot/data/huggingface/my-neural-network-data/transformer-from-scratch/tokenizers/wmt19-short/tokenizer_zh_jieba.json")

# count = 0
# for sentence in train.get_all_sentences(dataset_dict, "train", "zh", limit=None, verbose=False):
#     count += 1
# print(f"count = {count}")

# tokenizer = JiebaTokenizer()
# tokenizer.train_from_iterator(train.get_all_sentences(dataset_dict, "train", "zh", limit=None, verbose=False), min_frequency=2)
# tokenizer.save(tokenizer_path)

tokenizer = JiebaTokenizer.from_file(tokenizer_path)

print(f"vocab size: {tokenizer.get_vocab_size()}")

# strs = [
#     "当然，雷曼兄弟公司的倒闭和柏林墙的倒塌没有任何关系。",
#     "然而，和1989年一样，2008-2009年很可能也能被视为一个划时代的改变，其带来的发人深省的后果将在几十年后仍能让我们感受得到。",
#     "1989年，自由民主战胜了由苏联集团具体化并推崇的社会主义意识形态。",
# ]
# for s in strs:
#     encoding = tokenizer.encode(s, add_special_tokens=True)
#     print(s)
#     utils.print_encoding(encoding)
#     decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
#     print(f"decoded: {decoded}")
#     print()

# tok = tokenizer.id_to_token(1401720)
# print(tok)
#
# for norm_form in ["NFC", "NFD", "NFKD", "NFKC"]:
#     print(norm_form + ": " + unicodedata.normalize("NFC", tok))
