import os
from pathlib import Path
from typing import Dict, Any

ENV_LOCAL = "local"
ENV_LAMBDA = "lambda"

TRACKED = "tracked"
UNTRACKED = "untracked"

PROJECT_ROOT = Path(__file__).parent.resolve()

DATA_ROOTS = {
    ENV_LOCAL: {
        TRACKED: Path("/home/taot/data/huggingface/my-neural-network-data/transformer-from-scratch"),
        UNTRACKED: Path("/home/taot/data/ml_data/my_projects/transformer-from-scratch/local")
    },
    ENV_LAMBDA: {
        TRACKED: Path("/home/ubuntu/my-neural-network-data/transformer-from-scratch"),
        UNTRACKED: Path("/home/ubuntu/data")
    }
}


def get_config(env: str = ENV_LOCAL) -> Dict[str, Any]:
    tracked_data_root = DATA_ROOTS[env][TRACKED]
    untracked_data_root = DATA_ROOTS[env][UNTRACKED]

    if not tracked_data_root.exists() or not tracked_data_root.is_dir():
        raise Exception(f"{tracked_data_root} does not exist or is not a directory")

    untracked_data_root.mkdir(parents=True, exist_ok=True)

    return {
        "env": env,
        "tokenizer_file": "/home/taot/src_repos/pytorch-transformer/tokenizer_{0}.json",
        # "tokenizer_file": str(tracked_data_root) + "/tokenizers/tokenizer_{0}.json",
        "lang_src": "en",
        "lang_tgt": "it",
        "seq_len": 350,
        "dataset": "librakevin/opus_books_split",
        "dataset_config_name": "en-it",
        "batch_size": 8,
        "d_model": 512,
        "n_layers": 6,
        "num_epochs": 10,
        "validation_every_n_steps": -1,
        "validation_num_examples": 4,
        "lr": 10 ** -4,
        "preload": None,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "tensorboard_log_dir": str(untracked_data_root) + "/runs/tmodel"
    }

# batch_size: 8
# d_model: 512
# num_epochs: 20


def get_model_folder_path(config: Dict[str, Any]) -> Path:
    env = config["env"]
    untracked_data_root = DATA_ROOTS[env][UNTRACKED]

    model_folder_path = untracked_data_root / config["model_folder"]

    if model_folder_path.is_file():
        raise Exception(f"{model_folder_path} is a file instead of directory")

    if not model_folder_path.exists():
        print(f"Creating directory {model_folder_path}")
        model_folder_path.mkdir(exist_ok=True, parents=True)

    return model_folder_path


def get_weights_file_path(config: Dict[str, Any], epoch: int) -> Path:
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch:02d}.pt"
    return get_model_folder_path(config) / model_filename
