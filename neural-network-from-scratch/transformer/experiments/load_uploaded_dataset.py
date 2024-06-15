import datasets

dataset = datasets.load_dataset("librakevin/wmt19-short", name="zh-en-50", split="train")

print(dataset)
