import datasets

import train
from jieba_tokenizer_plain_py import JiebaTokenizer

SEQUENCE1 = "然而，和1989年一样，2008-2009年很可能也能被视为一个划时代的改变，其带来的发人深省的后果将在几十年后仍能让我们感受得到。"


def test_jieba_tokenizer() -> None:
    dataset_dict = datasets.load_dataset("wmt/wmt19", "zh-en")

    tokenizer_path = "/tmp/test_jieba_tokenizer_zh.json"

    tokenizer = JiebaTokenizer()
    tokenizer.train_from_iterator(train.get_all_sentences(dataset_dict, "train", "zh", limit=100, verbose=False))
    tokenizer.save(tokenizer_path)

    expected_tokens = [
        '[SOS]', '然而', '，', '和', '1989', '年', '一样', '，', '2008', '-', '2009', '年', '很', '可能', '也', '能',
        '被', '视为', '一个', '划时代', '的', '改变', '，', '其', '带来', '的', '发人深省', '的', '后果', '将', '在',
        '几十年', '后', '仍', '能', '让', '我们', '感受', '得到', '。', '[EOS]'
    ]

    encoding = tokenizer.encode(SEQUENCE1, add_special_tokens=True)
    assert encoding.tokens == expected_tokens
    decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
    assert decoded == f"[SOS]{SEQUENCE1}[EOS]"

    tokenizer = JiebaTokenizer.from_file(tokenizer_path)

    encoding = tokenizer.encode(SEQUENCE1, add_special_tokens=True)
    assert encoding.tokens == expected_tokens
    decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
    assert decoded == f"[SOS]{SEQUENCE1}[EOS]"
