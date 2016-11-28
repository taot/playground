# 数据集下载: http://yann.lecun.com/exdb/mnist/
# 读入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data.untracked/", one_hot=True)

import tensorflow as tf
