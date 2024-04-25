# 数据集下载: http://yann.lecun.com/exdb/mnist/
# 读入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data.untracked/", one_hot=True)

import tensorflow as tf
import mnist_image

# 实现回归模型

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 训练模型

# 正确值
y_ = tf.placeholder("float", [None, 10])

# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化变量
init = tf.initialize_all_variables()

# 创建 session，初始化
sess = tf.Session()
sess.run(init)

# 训练模型，循环 1000 次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 画出图像
import drawchar
drawchar.drawchar(mnist.test.images[0])

# 运行神经网络
pred = tf.argmax(y, 1)
results = sess.run(pred, feed_dict={x: mnist.test.images})

# 检查结果是否匹配
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels }))
