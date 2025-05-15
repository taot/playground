import math
from contextlib import nullcontext

from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer  # type: ignore[import-untyped]

from config import LMConfig, PretrainConfig
from datasets import PretrainDataset
from model.model import MiniTransformer


def init_model(config: LMConfig):
    print(f"Config: {config}")

    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    print(f"Vocab size: {tokenizer.vocab_size}")

    model = MiniTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_seq=config.n_seq,
        h=config.h,
        n_layers=config.n_layers,
        dropout=config.dropout
    )
    print(f"LLM 总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万")

    return model, tokenizer


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


class PreTrainer:

    def __init__(self, lm_config: LMConfig, pretrain_config: PretrainConfig) -> None:
        self.lm_config = lm_config
        self.pretrain_config = pretrain_config

        self.model, self.tokenizer = init_model(lm_config)

        train_ds = PretrainDataset(pretrain_config.data_path, self.tokenizer, max_length=lm_config.n_seq)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=pretrain_config.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            sampler=None
        )

        self.optimizer = optim.AdamW(self.model.parameters(), lr=pretrain_config.learning_rate)

        self.context = nullcontext()

    def train(self) -> None:
        for epoch in range(self.pretrain_config.n_epochs):
            self._train_epoch(epoch)

    def _train_epoch(self, epoch: int) -> None:
        loss_fn = nn.CrossEntropyLoss(reduction="none")

        iter_per_epoch = len(self.train_loader)

        for step, (X, Y, loss_mask, texts) in enumerate(self.train_loader):
            # X, Y: (batch_size, n_seq): int
            # loss_mask: (batch_size, n_seq): bool
            lr = get_lr(epoch * iter_per_epoch + step, self.pretrain_config.n_epochs * iter_per_epoch, self.pretrain_config.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            with self.context:
                result = self.model(X)  # result: (batch_size, n_seq, vocab_size)
                flat_result = result.view(-1, result.shape[-1])     # flat_result: (batch_size * n_seq, vocab_size)
                flat_Y = Y.view(-1)     # flat_Y: (batch_size * n_seq)
                loss = loss_fn(flat_result, flat_Y).view(Y.shape)   # loss: (batch_size, n_seq)
                loss = (loss * loss_mask).sum() / loss_mask.sum()

                print(f"in loop {step}: lr = {lr}, loss = {loss}")


def main() -> None:
    lm_config = LMConfig(
        d_model=128,
        n_seq=500,
        h=8,
        n_layers=8,
        dropout=0.1,
    )

    pretrain_config = PretrainConfig(
        data_path="/home/taot/data/ml_data/my_projects/minimind/dataset/pretrain_hq.jsonl",
        batch_size=2, # 32
        learning_rate=5e-4,
        n_epochs=1
    )

    pretrainer = PreTrainer(lm_config, pretrain_config)
    pretrainer.train()


if __name__ == "__main__":
    main()
