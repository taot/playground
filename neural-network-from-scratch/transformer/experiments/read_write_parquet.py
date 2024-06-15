import pyarrow as pa
import pyarrow.parquet as pq

table = pq.read_table("/home/taot/data/ml_data/my_projects/experiments/train-00000-of-00013.parquet")

