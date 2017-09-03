import nn; reload(nn)
import numpy as np

import mnist_loader; reload(mnist_loader)
import mnist_image; reload(mnist_image)

train, valid, test = mnist_loader.load_data()

network = nn.Network([784, 20, 10])

network.forward(train[0][0].reshape((784,1)))

print network.A[0].shape

y = train[1][0]
y_oh = nn.onehot(y)

network.backprop(y_oh)
