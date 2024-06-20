import train
from config import get_config

config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = train.get_ds(config)

for i in range(3):
    print(f"\n\nIteration #{i}")
    count = 0
    for batch in val_dataloader:
        if count > 2:
            break

        print(batch["src_text"][0])
        count += 1
