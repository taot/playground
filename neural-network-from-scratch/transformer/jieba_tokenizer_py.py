import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, List, Iterable, Dict, Optional, Iterator

import jieba

UNK = "[UNK]"
PAD = "[PAD]"
SOS = "[SOS]"
EOS = "[EOS]"

SPECIAL_TOKENS = [UNK, PAD, SOS, EOS]


@dataclass
class JiebaEncoding:
    ids: List[int]
    tokens: List[str]
    special_tokens_mask: List[int]

    def __post_init__(self):
        assert len(self.ids) == len(self.tokens) and len(self.ids) == len(self.special_tokens_mask)


class JiebaTokenizer:

    def __init__(self):
        super().__init__()
        self.__data = []    # list of tuple (token, id, frequency)
        self.__token_to_id_map = {}
        self.__id_to_token_map = {}

    @staticmethod
    def from_file(file: str | Path) -> 'JiebaTokenizer':
        with open(file, "r", encoding="utf-8") as fp:
            json_body = json.load(fp)

        tokenizer = JiebaTokenizer()
        tokenizer.__data = json_body["data"]
        tokenizer.__reconstruct_from_data()

        return tokenizer

    def save(self, file: str | Path) -> None:
        json_body = {
            "data": self.__data
        }
        with open(file, "w", encoding="utf-8") as fp:
            json.dump(json_body, fp, indent=4, ensure_ascii=False)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> JiebaEncoding:
        sequence = self.normalize(sequence)
        pre_tokens = self.pre_tokenize(sequence)
        unk_id = self.token_to_id(UNK)

        ids = []
        tokens = []
        special_tokens_mask = []

        for pre_tok in pre_tokens:
            id = self.token_to_id(pre_tok)
            if id is None:
                ids.append(unk_id)
                tokens.append(UNK)
                special_tokens_mask.append(1)
            else:
                ids.append(id)
                tokens.append(pre_tok)
                special_tokens_mask.append(0)

        encoding = JiebaEncoding(ids, tokens, special_tokens_mask)
        encoding = self.post_process(encoding, add_special_tokens)

        return encoding

    def encode_batch(self, sequences: Iterable[str], add_special_tokens: bool = True) -> List[JiebaEncoding]:
        return [self.encode(s, add_special_tokens) for s in sequences]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        special_token_ids = {self.__token_to_id_map[t] for t in SPECIAL_TOKENS}
        tokens = [self.id_to_token(id) for id in ids if not (skip_special_tokens and id in special_token_ids)]
        return "".join(tokens)

    def decode_batch(self, sequences: Iterable[Iterable[int]], skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(ids, skip_special_tokens) for ids in sequences]

    def get_vocab(self) -> Dict[str, int]:
        return self.__token_to_id_map

    def id_to_token(self, id: int) -> Optional[str]:
        return self.__id_to_token_map.get(id)

    def token_to_id(self, token: str) -> Optional[int]:
        return self.__token_to_id_map.get(token)

    def train_from_iterator(self, iterator: Iterator[str], min_frequency: Optional[int] = None) -> None:
        counter = Counter()
        for sequence in iterator:
            sequence = self.normalize(sequence)
            tokens = self.pre_tokenize(sequence)
            counter.update(tokens)

        if min_frequency is not None and min_frequency > 0:
            counter = Counter({k: c for k, c in counter.items() if c >= min_frequency})

        sorted_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        self.__data = []
        id = 0

        # Special tokens
        for token in SPECIAL_TOKENS:
            self.__data.append((token, id, -1))
            id += 1

        # Trained tokens
        for token, freq in sorted_items:
            self.__data.append((token, id, freq))
            id += 1

        self.__reconstruct_from_data()

    def normalize(self, sequence: str) -> str:
        return unicodedata.normalize("NFC", sequence)

    def pre_tokenize(self, sequence: str) -> List[str]:
        def is_empty_string(s: Optional[str]) -> bool:
            return s is None or len(s.strip()) == 0

        tokens = jieba.cut(sequence, cut_all=False)
        tokens = [t for t in tokens if not is_empty_string(t)]
        return tokens

    def post_process(self, encoding: JiebaEncoding, add_special_tokens=True) -> JiebaEncoding:
        if not add_special_tokens:
            return encoding

        ids = [self.token_to_id(SOS)] + encoding.ids + [self.token_to_id(EOS)]
        tokens = [SOS] + encoding.tokens + [EOS]
        special_tokens_mask = [1] + encoding.special_tokens_mask + [1]

        return JiebaEncoding(ids, tokens, special_tokens_mask)

    def __reconstruct_from_data(self):
        self.__token_to_id_map = {token: id for token, id, _ in self.__data}
        self.__id_to_token_map = {id: token for token, id, _ in self.__data}
