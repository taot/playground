import datasets

dataset = datasets.load_dataset("/home/taot/data/ml_data/my_projects/experiments/wmt19-small", split="train")

print(dataset)
