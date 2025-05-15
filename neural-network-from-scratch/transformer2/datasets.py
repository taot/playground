import json

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        text = str(sample["text"])
        add_bos = self.tokenizer.bos_token if not text.startswith(self.tokenizer.bos_token) else ""
        add_eos = self.tokenizer.eos_token if not text.endswith(self.tokenizer.eos_token) else ""
        text = f"{add_bos}{str(sample['text'])}{add_eos}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length + 1,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # X = torch.tensor(input_ids[:-1], dtype=torch.long)
        # Y = torch.tensor(input_ids[1:], dtype=torch.long)
        # loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        X = input_ids[:-1].clone().detach()
        Y = input_ids[1:].clone().detach()

        # TODO why loss_mask[1:] ?
        loss_mask = loss_mask[1:].clone().detach()

        return X, Y, loss_mask, text
