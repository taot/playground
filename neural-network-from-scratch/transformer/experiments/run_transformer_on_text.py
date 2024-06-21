import torch
from torch.utils.tensorboard import SummaryWriter

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

batch = None

for b in val_dataloader:
    batch = b
    break

encoder_input = batch["encoder_input"].to(device)
encoder_mask = batch["encoder_mask"].to(device)

assert encoder_input.size(0) == 1
assert encoder_mask.size(0) == 1

model_out = train.greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, config["seq_len"], device)
model_out_np = model_out.detach().cpu().numpy()
# ids = model_out_np[0]
model_out_text = tokenizer_tgt.decode(model_out_np, skip_special_tokens=False)

source_text = batch["src_text"][0]
target_text = batch["tgt_text"][0]

# Print to console
print(f"SOURCE: {source_text}")
print(f"TARGET: {target_text}")
print(f"PREDICTED: {model_out_text}")


writer = SummaryWriter('runs/transformer_1')

writer.add_graph(model, model_out)
writer.close()
