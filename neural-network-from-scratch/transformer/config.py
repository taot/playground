import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

config = {
    "tokenizer_file": str(PROJECT_ROOT) + "/tokenizers/tokenizer_{0}.json",
}
