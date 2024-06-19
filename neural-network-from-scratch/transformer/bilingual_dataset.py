from typing import Any

import torch

from tokenizers import Tokenizer
from torch.utils.data import Dataset

from constants import SOS, EOS, PAD


def causal_mask(size: int):
    mask = torch.triu(torch.ones(1, size, size, dtype=torch.int), diagonal=1)
    mask = mask == 0
    return mask


class BilingualDataset(Dataset):

    def __init__(self, ds: Dataset, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, lang_src: str, lang_tgt: str, seq_len: int) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id(SOS)], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id(EOS)], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id(PAD)], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair["translation"][self.lang_src]
        tgt_text = src_tgt_pair["translation"][self.lang_tgt]

        enc_input_token_ids = self.tokenizer_src.encode(src_text, add_special_tokens=False).ids
        dec_input_token_ids = self.tokenizer_src.encode(src_text, add_special_tokens=False).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_token_ids) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_token_ids) - 1

        if enc_num_padding_tokens < 0:
            raise ValueError(f"Text too long, index = {index}, lang = {self.lang_src}, seq_len = {self.seq_len}: {src_text}")

        if dec_num_padding_tokens < 0:
            raise ValueError(f"Text too long, index = {index}, lang = {self.lang_tgt}, seq_len = {self.seq_len}: {tgt_text}")

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_token_ids, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_token_ids, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ])

        label = torch.cat([
            torch.tensor(enc_input_token_ids, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()    # (1, 1, seq_len)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() \
                       & causal_mask(decoder_input.size(0))     # (1, seq_len) & (1, seq_len, seq_len)
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
