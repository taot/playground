import os
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.resolve()

PROJECT_DATA_ROOT = Path("/home/taot/data/ml_data/my_projects/transformer_from_scratch")


def get_config() -> Dict[str, Any]:
    return {
        "tokenizer_file": str(PROJECT_DATA_ROOT) + "/tokenizers/tokenizer_{0}.json",
        "lang_src": "en",
        "lang_tgt": "zh",
        "seq_len": 52,
        "dataset": "librakevin/wmt19-short",
        "dataset_config_name": "zh-en-50",
        "batch_size": 8,
        "d_model": 512,
        "n_layers": 6,
        "num_epochs": 20,
        "lr": 10**-4,
        "preload": None,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "tensorboard_log_dir": str(PROJECT_DATA_ROOT) + "/runs/tmodel"
    }

# batch_size: 8
# d_model: 512
# num_epochs: 20

def get_model_folder_path(config: Dict[str, Any]) -> Path:
    return PROJECT_DATA_ROOT / config["model_folder"]


def get_weights_file_path(config: Dict[str, Any], epoch: int) -> Path:
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch:02d}.pt"
    return get_model_folder_path(config) / model_filename
