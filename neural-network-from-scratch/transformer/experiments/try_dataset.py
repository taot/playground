import os

import datasets


hf_home = os.environ["HF_HOME"]
print(hf_home)

# inspect_dataset("wmt19", "path/to/scripts")
# builder = load_dataset_builder(
#     "wmt/wmt19",
#     # language_pair=("en", "zh"),
#     # subsets={
#     #     datasets.Split.TRAIN: ["commoncrawl_frde"],
#     #     datasets.Split.VALIDATION: ["euelections_dev2019"],
#     # },
# )
#
# # Standard version
# builder.download_and_prepare()
# ds = builder.as_dataset()
#
# # Streamable version
# ds = builder.as_streaming_dataset()

ds = datasets.load_dataset("wmt/wmt19", "zh-en")
