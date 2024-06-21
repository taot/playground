import torch

import train
from config import get_config, ENV_LOCAL, get_weights_file_path

device = "cpu"
config = get_config(env=ENV_LOCAL)

count = 0

train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = train.get_ds(config)
model = train.get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to("cpu")

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

model_weights_path = "/home/taot/data/ml_data/my_projects/pytorch-transformer/20240619/opus_books_weights/tmodel_19.pt"
initial_epoch, global_step = train.load_model_state(model, optimizer, model_weights_path, map_location=device)
print(f"initial_epoch: {initial_epoch}, global_step: {global_step}")


train.run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len=config["seq_len"], device=device, print_msg=lambda msg: print(msg), num_examples=4)
