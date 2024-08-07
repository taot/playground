import warnings
from typing import Dict, Any, Optional
import sys

import torch
from datasets import load_dataset
from jieba_tokenizer_py import JiebaTokenizer
from tokenizers.processors import TemplateProcessing
from torch import nn, Tensor

import datasets
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel

from pathlib import Path

from torch.optim import Optimizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import get_model_folder_path, get_weights_file_path, get_config, ENV_LOCAL, ENV_LAMBDA
from constants import PAD, SOS, EOS
from model import build_transformer, Transformer
from bilingual_dataset import BilingualDataset, causal_mask


def get_all_sentences(dataset: datasets.Dataset, lang: str, *, limit: Optional[int] = None, verbose: bool = False):
    if limit is not None:
        assert limit >= 0, "limit must be greater than or equal to 0"
        ds = dataset.select(range(limit))

    with tqdm(total=len(dataset)) as progress:
        for item in dataset:
            sentence = item["translation"][lang]
            if verbose:
                print(sentence)
            progress.update(1)
            yield sentence


def get_tokenizer(config: Dict[str, Any], lang: str) -> Optional[Tokenizer]:
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    print(f"Getting tokenizer from {tokenizer_path}")

    if not tokenizer_path.exists():
        return None

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def build_bbpe_tokenizer(config: Dict[str, Any], dataset: datasets.Dataset, lang: str) -> Tokenizer:
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    print(f"tokenizer_path = {tokenizer_path}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2, show_progress=True)

    tokenizer.train_from_iterator(iterator=get_all_sentences(dataset, lang, limit=None, verbose=False), trainer=trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.save(str(tokenizer_path))

    return tokenizer


def build_word_level_tokenizer(config: Dict[str, Any], dataset: datasets.Dataset, lang: str):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if lang == "zh":
        tokenizer = JiebaTokenizer()
        tokenizer.train_from_iterator(get_all_sentences(dataset, "zh", limit=None, verbose=False), min_frequency=2)
        tokenizer.save(tokenizer_path)

    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        tokenizer.train_from_iterator(iterator=get_all_sentences(dataset, lang, limit=None, verbose=False), trainer=trainer)

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
    assert tokenizer_src is not None
    assert tokenizer_tgt is not None

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config: Dict[str, Any], src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    seq_len = config["seq_len"]
    model = build_transformer(src_vocab_size, tgt_vocab_size, seq_len, seq_len, config["d_model"], N=config["n_layers"])
    return model


def greedy_decode(model: Transformer, source: Tensor, src_mask: Tensor, tokenizer_src: Tokenizer, max_len: int, device: str) -> Tensor:
    sos_id = tokenizer_src.token_to_id(SOS)
    eos_id = tokenizer_src.token_to_id(EOS)

    encoder_output = model.encode(source, src_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_id).type_as(source).to(device)

    while True:
        if decoder_input.size(1) >= max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_id:
            break

    return decoder_input.squeeze(0)


def run_validation(model: Transformer, val_ds: BilingualDataset, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer,
                   max_len: int, device: str, print_msg, num_examples=2) -> None:
    model.eval()
    count = 0
    print_msg(f"num_examples = {num_examples}")

    with torch.no_grad():
        for batch in val_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1
            assert encoder_mask.size(0) == 1

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, max_len, device)
            model_out_np = model_out.detach().cpu().numpy()
            # ids = model_out_np[0]
            model_out_text = tokenizer_tgt.decode(model_out_np, skip_special_tokens=False)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            # Print to console
            print_msg("-" * 80)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count >= num_examples:
                break


def load_model_state(model: Transformer, optimizer: Optimizer, model_weightw_path: str, map_location=None) -> int:
    print(f"Preloading model {model_weightw_path}")
    state = torch.load(model_weightw_path, map_location=map_location)
    model.load_state_dict(state['model_state_dict'])
    initial_epoch = state["epoch"] + 1
    optimizer.load_state_dict(state["optimizer_state_dict"])
    global_step = state["global_step"]

    return initial_epoch, global_step


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
    if config["preload"] is not None:
        model_weights_path = get_weights_file_path(config, config["preload"])
        initial_epoch, global_step = load_model_state(model, optimizer, model_weights_path)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(PAD), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        count = 0
        for batch in batch_iterator:
            count += 1
            # if count > 2:
            #     break

            model.train()

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
            optimizer.zero_grad(set_to_none=True)       # Try `set_to_none=False`

            validation_every_n_steps = config["validation_every_n_steps"]
            if validation_every_n_steps > 0 and global_step % config["validation_every_n_steps"] == 0:
                run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device,
                               lambda msg: batch_iterator.write(msg), num_examples=config["validation_num_examples"])

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg),
                       num_examples=config["validation_num_examples"])

        # Save the model
        model_file_path = get_weights_file_path(config, epoch)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_file_path)


def main():
    # warnings.filterwarnings("ignore")

    env = ENV_LOCAL
    if len(sys.argv) >= 2:
        env = sys.argv[1]
        if env not in [ENV_LOCAL, ENV_LAMBDA]:
            raise Exception(f"Invalid env. Valid values are {ENV_LOCAL} and {ENV_LAMBDA}")

    config = get_config(env)
    train_model(config)


if __name__ == "__main__":
    main()
