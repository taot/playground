import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers, processors, decoders
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

import train
import utils

tokenizer_path = "/home/taot/tmp/tokenizers/tokenizer_zh.json"

vocab_size = int(50000)
min_freq = 2


def train_tokenizer() -> Tokenizer:
    dataset = datasets.load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train")
    data_iterator = train.get_all_sentences(dataset, "zh", limit=None, verbose=False)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=min_freq
    )
    # tokenizer.post_processor = processors.ByteLevel()

    tokenizer.decoder = decoders.BPEDecoder()

    tokenizer.train_from_iterator(data_iterator, trainer=trainer)



    # tokenizer.post_processor = TemplateProcessing(
    #     single="[SOS] $0 [EOS]",
    #     pair="[SOS] $A [EOS] $B:1 [EOS]:1",
    #     special_tokens=[
    #         ("[SOS]", tokenizer.token_to_id("[SOS]")),
    #         ("[EOS]", tokenizer.token_to_id("[EOS]"))
    #     ],
    # )

    tokenizer.save(tokenizer_path)

    return tokenizer


def run_tokenizer() -> None:
    tokenizer = Tokenizer.from_file(tokenizer_path)

    sentence = "然而，和1989年一样，2008-2009年很可能也能被视为一个划时代的改变，其带来的发人深省的后果将在几十年后仍能让我们感受得到。"
    encoding = tokenizer.encode(sentence, add_special_tokens=True)
    utils.print_encoding(encoding)
    print(tokenizer.decode(encoding.ids, skip_special_tokens=True))


# train_tokenizer()
run_tokenizer()
