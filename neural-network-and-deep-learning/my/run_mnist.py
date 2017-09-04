import nn; reload(nn)
import numpy as np

import mnist_loader; reload(mnist_loader)
import mnist_image; reload(mnist_image)

train, valid, test = mnist_loader.load_data_wrapper()

train_size = len(train)

# def get_reshape_input_vector(dataset, i):
#     x = dataset[0][i]
#     r = x.reshape((784, 1))
#     return r
#
#
# def get_output_vector(dataset, i):
#     y = dataset[1][i]
#     return nn.onehot(y)

network = nn.Network([784, 100, 10])


def validate(net, n_epoch, validation_set):
    total_count = len(validation_set)
    count = 0
    for v in validation_set:
        x = v[0]
        y = v[1]
        net.forward(x)
        y0_v = net.get_outputs()
        y0 = np.argmax(y0_v)
        if y0 == y:
            count += 1

    print("epoch: %d, validation rate: %f" % (n_epoch, 1.0 * count / total_count))


def epoch(net, n_epoch):
    np.random.shuffle(train)
    err_sum = 0.0
    for i in range(0, train_size):
        x = train[i][0]
        y = train[i][1]
        net.forward(x)
        err = nn.mse(y, network.get_outputs())
        err_sum += err
        net.backprop(y)
    print("epoch: %d, error: %f" % (n_epoch, err_sum / train_size))
    validate(net, n_epoch, valid)


print("Start training...")

network.lr = 1e-4
for i in range(0, 10):
    epoch(network, i)

network.lr = 5e-5
for i in range(10, 20):
    epoch(network, i)

network.lr = 1e-5
for i in range(20, 40):
    epoch(network, i)
network.save()

network.lr = 5e-6
for i in range(40, 50):
    epoch(network, i)
network.save()

pass
