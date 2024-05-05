import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

PROJECT_DATA_ROOT = Path("/home/taot/data/ml_data/my_projects/transformer_from_scratch")

config = {
    # "tokenizer_file": str(PROJECT_ROOT) + "/tokenizers/tokenizer_{0}.json",
    "tokenizer_file": str(PROJECT_DATA_ROOT) + "/tokenizers/tokenizer_{0}.json",
}
