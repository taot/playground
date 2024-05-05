from jieba_tokenizer_py import JiebaEncoding
from tokenizers import Encoding


def print_encoding(encoding: Encoding | JiebaEncoding) -> None:
    print("ids: ", encoding.ids)
    print("tokens: ", encoding.tokens)
    print("special_tokens_mask: ", encoding.special_tokens_mask)
    if hasattr(encoding, "type_ids"):
        print("type_ids: ", encoding.type_ids)
    if hasattr(encoding, "offsets"):
        print("offsets: ", encoding.offsets)
