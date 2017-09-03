import numpy as np

def relu(x):
    return np.maximum(x, 0)

def relu_p(x):
    return np.maximum(np.sign(x), 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_p(x):
    return 1 / (1 + np.exp(x))

def mse(x, y):
    return np.square(x - y).sum() / 2

def mse_p(x, y):
    return x - y

def onehot(x, size):
    a = np.zeros((size, 1))
    a[x, 0] = 1
    return a

class Network:

    def __init__(self, dims):
        self.dims = dims
        self.n_layers = len(dims) - 1
        self.W = []     # weights
        self.B = []     # biases
        self.Z = []     # intermediate Z
        self.A = []     # activations
        self.lr = 1e-4  # learning rate

        # initialize weights and biases randomly
        for i in range(0, self.n_layers - 1):
            w = np.random.rand(dims[i+1], dims[i])
            self.W.append(w)
            b = np.random.rand(dims[i+1], 1)
            self.B.append(b)

    def get_input_dim(self):
        return self.dims[0]

    def get_output_dim(self):
        return self.dims[self.n_layers - 1]

    def get_outputs(self):
        return self.A[self.n_layers - 2]

    def forward(self, inputs):
        self.Z = []
        self.A = []
        self.inputs = inputs
        a = inputs
        for i in range(0, self.n_layers - 1):
            y = np.matmul(self.W[i], a)
            z = y + self.B[i]
            self.Z.append(z)
            if i < self.n_layers - 2:
                a = relu(z)
            else:
                a = sigmoid(z)
            self.A.append(a)

    def backprop(self, y):
        i = self.n_layers - 2
        delta = mse_p(A[i], y) * sigmoid_p(Z[i])
        self.update_weights(i, delta)
        for i in (self.n_layers - 3, -1, -1):
            delta = np.matmul(self.W[i+1].transpose(), delta) * relu_p(self.Z[i])
            self.update_weights(i, delta)

    def update_weights(self, l, delta):
        self.B -= (delta * self.lr)
        if l == 0:
            a = self.inputs
        else:
            a = A[l-1]
        self.W[l] -= (np.matmul(a, delta) * self.lr)
