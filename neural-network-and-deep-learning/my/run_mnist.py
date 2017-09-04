import nn; reload(nn)
import numpy as np

import mnist_loader; reload(mnist_loader)
import mnist_image; reload(mnist_image)

train, valid, test = mnist_loader.load_data()

train_size = train[0].shape[0]


def get_reshape_input_vector(dataset, i):
    return dataset[0][i].reshape((784, 1))


def get_output_vector(dataset, i):
    y = dataset[1][i]
    return nn.onehot(y)


network = nn.Network([784, 100, 10])
network.lr = 1e-4


def epoch(net, n_epoch):
    for i in range(0, train_size):
        x = get_reshape_input_vector(train, i)
        y = get_output_vector(train, i)
        net.forward(x)
        err = nn.mse(y, network.get_outputs())
        print("epoch: %d, iter: %d, error: %f" % (n_epoch, i, err))
        net.backprop(y)

for i in range(0, 2):
    epoch(network, i)

lr = 1e-6
for i in range(2, 4):
    epoch(network, i)

pass
