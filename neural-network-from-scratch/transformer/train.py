import warnings
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from jieba_tokenizer_py import JiebaTokenizer
from tokenizers.processors import TemplateProcessing
from torch import nn

import datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import get_model_folder_path, get_weights_file_path, get_config
from constants import PAD
from model import build_transformer, Transformer
from mydataset import BilingualDataset


def get_all_sentences(dataset_dict: datasets.DatasetDict, key: str, lang: str, *, limit: Optional[int] = None, verbose: bool = False):

    def print_log(s: str) -> None:
        print(f"get_all_training_sentences: {s}")

    assert key in dataset_dict.keys(), f"dataset key must be one of {dataset_dict.keys()}"
    print_log(f"key = {key}, lang = {lang}, limit = {limit}")

    ds = dataset_dict[key]

    if limit is not None:
        assert limit >= 0, "limit must be greater than or equal to 0"
        ds = ds.select(range(limit))

    with tqdm(total=len(ds)) as progress:
        for item in ds:
            sentence = item["translation"][lang]
            if verbose:
                print_log(sentence)
            progress.update(1)
            yield sentence


def get_tokenizer(config: Dict[str, Any], lang: str) -> Optional[Tokenizer]:
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not tokenizer_path.exists():
        return None

    if lang == "zh":
        tokenizer = JiebaTokenizer.from_file(tokenizer_path)
        return tokenizer
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer


def build_tokenizer(config: Dict[str, Any], dataset_dict: datasets.DatasetDict, lang: str):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if lang == "zh":
        tokenizer = JiebaTokenizer()
        tokenizer.train_from_iterator(get_all_sentences(dataset_dict, "train", "zh", limit=None, verbose=False), min_frequency=2)
        tokenizer.save(tokenizer_path)

    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        tokenizer.train_from_iterator(iterator=get_all_sentences(dataset_dict, "train", lang, limit=None, verbose=False), trainer=trainer)

        tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $0 [EOS]",
            pair="[SOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", tokenizer.token_to_id("[SOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]"))
            ],
        )

        tokenizer.save(str(tokenizer_path))


def get_or_build_tokenizer(config: Dict[str, Any], dataset_dict: datasets.DatasetDict, lang: str) -> Tokenizer:
    tokenizer = get_tokenizer(config, lang)
    if tokenizer is not None:
        return tokenizer

    build_tokenizer(config, dataset_dict, lang)
    tokenizer = get_tokenizer(config, lang)

    return tokenizer


def get_ds(config: Dict[str, Any]) -> (DataLoader, DataLoader, Tokenizer, Tokenizer):
    dataset_name = config["dataset"]
    dataset_config_name = config["dataset_config_name"]
    train_ds_raw = load_dataset(dataset_name, dataset_config_name, split="train")
    val_ds_raw = load_dataset(dataset_name, dataset_config_name, split="validation")

    # Build tokenizers
    tokenizer_src = get_tokenizer(config, config["lang_src"])
    tokenizer_tgt = get_tokenizer(config, config["lang_tgt"])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config: Dict[str, Any], src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    seq_len = config["seq_len"]
    model = build_transformer(src_vocab_size, tgt_vocab_size, seq_len, seq_len, config["d_model"], N=config["n_layers"])
    return model


def train_model(config: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    get_model_folder_path(config).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    tensorboard_writer = SummaryWriter(config["tensorboard_log_dir"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_path = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_path}")
        state = torch.load(model_path)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(PAD), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)   # (B, seq_len)
            decoder_input = batch["decoder_input"].to(device)   # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)     # (B, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)     # (B, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)     # (B, seq_len, tgt_vocab_size)

            label = batch["label"].to(device)   # (B, seq_len)

            # (B, seq_len, tgt_vocab_size) -> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log loss
            tensorboard_writer.add_scalar("train_loss", loss.item(), global_step)
            tensorboard_writer.flush()

            # Back-propagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save the model
        model_file_path = get_weights_file_path(config, epoch)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        })


def main():
    # warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)


if __name__ == "__main__":
    main()
