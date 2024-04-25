# 数据集下载: http://yann.lecun.com/exdb/mnist/
# 读入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data.untracked/", one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())
y = tf.matmul(x, W) + b
