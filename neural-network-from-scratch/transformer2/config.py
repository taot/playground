from dataclasses import dataclass


@dataclass
class LMConfig:
    d_model: int
    n_seq: int
    h: int
    n_layers: int
    dropout: float = 0.1


@dataclass
class PretrainConfig:
    data_path: str
    batch_size: int
    learning_rate: float
    n_epochs: int
