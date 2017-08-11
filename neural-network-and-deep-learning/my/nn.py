import numpy as np

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(a, b):
    return np.square(a - b).sum() / 2

class Network:

    def __init__(self, dims):
        self.dims = dims
        self.n_layers = len(dims) - 1
        self.W = []
        self.B = []
        for i in range(0, self.n_layers):
            w = np.random.rand(dims[i+1], dims[i])
            self.W.append(w)
            b = np.random.rand(dims[i+1], 1)
            self.B.append(b)

    def get_input_dim(self):
        return self.dims[0]

    def get_output_dim(self):
        return self.dims[self.n_layers - 1]

    def forward(self, inputs):
        a = inputs
        for i in range(0, self.n_layers):
            y = np.matmul(self.W[i], a)
            if i < self.n_layers - 1:
                a = relu(y + self.B[i])
            else:
                a = sigmoid(y + self.B[i])
        return a
