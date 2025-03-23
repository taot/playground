from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


def get_text() -> str:
    DATA_FILE = "/home/taot/data/huggingface/my-neural-network-data/gpt-from-scratch/tiny_shakespeare.txt"
    with open(DATA_FILE, "r") as fp:
        text = fp.read()
    print(len(text))
    return text


class CharTokenizer:

    def __init__(self, vocab: List[str]):
        super().__init__()
        self.vocab = vocab
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for i, c in enumerate(vocab)}

    @staticmethod
    def from_data_text(text: str) -> 'CharTokenizer':
        vocab = sorted(set(text))
        print(f"vocab_size = {len(vocab)}")
        print(f"vocab = {''.join(vocab)}")
        return CharTokenizer(vocab)

    def get_vocab(self) -> List[str]:
        return self.vocab

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.itos[i] for i in ids])


def get_train_val_data(text: str, tokenizer: CharTokenizer, device: torch.device) -> (Tensor, Tensor):
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    # Split the data into train and validation
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def get_batch(data: Tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, max_new_tokens: int, embed_dim: int = 32):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding_table = nn.Embedding(max_new_tokens, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None):
        device = idx.device
        B, T = idx.size()
        # idx and targets are both (B,T). B is batch_size, T is time (block_size), C is classes
        tok_embed = self.token_embedding_table(idx)  # (B, T, embed_dim)
        pos_embed = self.pos_embedding_table(torch.arange(T, device=device))    # (T, embed_dim)
        x = tok_embed + pos_embed   # broadcast
        logits = self.lm_head(x)    # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss  # (B,T,C)

    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        # idx: (B, T)
        print(f"generate max_new_token = {max_new_tokens}")
        for i in range(max_new_tokens):
            # print(f"generate iter {i}")
            logits, _ = self(idx)  # (B, T, C)
            logits = logits[:, -1, :]  # (B, 1, C)

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

        return idx


BLOCK_SIZE = 8
BATCH_SIZE = 32
TRAIN_ITERS = 10
EVAL_ITERS = 100
EVAL_INTERVAL = 50
MAX_NEW_TOKENS = 500


def train(model: BigramLanguageModel, train_data: Tensor, val_data: Tensor, eval_interval: int):
    model.train()
    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for iter in range(TRAIN_ITERS):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, eval_iters=EVAL_ITERS, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
            print(f"iter {iter}: train_loss = {losses['train']}, val_loss = {losses['val']}")

        xb, yb = get_batch(train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
        targets, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def estimate_loss(model: BigramLanguageModel, train_data: Tensor, val_data: Tensor, eval_iters: int, block_size: int, batch_size: int):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size=block_size, batch_size=batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


def main() -> None:
    torch.manual_seed(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = get_text()
    tokenizer = CharTokenizer.from_data_text(text)
    train_data, val_data = get_train_val_data(text, tokenizer, device=device)
    model = BigramLanguageModel(vocab_size=tokenizer.get_vocab_size(), max_new_tokens=MAX_NEW_TOKENS)
    model.to(device=device)

    train(model, train_data, val_data, EVAL_INTERVAL)

    # eval(model, val_data)

    generated_text = tokenizer.decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=MAX_NEW_TOKENS)[0].tolist())
    print(f"generated_text: {generated_text}")


if __name__ == "__main__":
    main()
