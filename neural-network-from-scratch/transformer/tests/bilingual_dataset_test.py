import datasets
import torch
from tokenizers import Tokenizer

import bilingual_dataset
import train
from config import get_config
from bilingual_dataset import BilingualDataset


def test_causal_mask() -> None:
    mask = bilingual_dataset.causal_mask(4)
    assert mask[0, 0, 0]
    assert not mask[0, 0, 1]
    assert mask[0, 3, 0]
    assert mask[0, 3, 3]


# def test_bilingual_dataset() -> None:
#     config = get_config()
#     seq_len = 50
#     ds = datasets.load_dataset("wmt/wmt19", f'zh-en', split="train")
#     tokenizer_src = train.get_tokenizer(config, "en")
#     tokenizer_tgt = train.get_tokenizer(config, "zh")
#     blds = BilingualDataset(ds, tokenizer_src,tokenizer_tgt, "en", "zh", seq_len)
#     assert len(blds) > 0
#
#     item = blds[1]
#     assert item["encoder_input"].size() == torch.Size([seq_len])
#     assert item["decoder_input"].size() == torch.Size([seq_len])
#     assert item["label"].size() == torch.Size([seq_len])
#     assert item["encoder_mask"].size() == torch.Size([1, 1, seq_len])
#     assert item["decoder_mask"].size() == torch.Size([1, seq_len, seq_len])
