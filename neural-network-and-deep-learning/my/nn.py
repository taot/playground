import numpy as np
import os
import datetime
import pickle

def relu(x):
    return np.maximum(x, 0)

def relu_p(x):
    return np.maximum(np.sign(x), 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_p(x):
    # return 1 / (1 + np.exp(x))
    return sigmoid(x) * (1 - sigmoid(x))

def mse(x, y):
    return np.square(x - y).sum() / 2

def mse_p(x, y):
    return x - y

def onehot(x):
    a = np.zeros((10, 1))
    a[x, 0] = 1
    return a

class Network:

    save_path = "/home/taot/tmp/neural-network-and-deep-learning/"

    def __init__(self, dims):
        self.dims = dims
        self.n_layers = len(dims)
        self.W = []     # weights
        self.B = []     # biases
        self.Z = []     # intermediate Z
        self.A = []     # activations
        self.lr = 1e-5  # learning rate

        # initialize weights and biases randomly
        for i in range(0, self.n_layers - 1):
            w = (np.random.rand(dims[i+1], dims[i]) / 25)
            self.W.append(w)
            b = (np.random.rand(dims[i+1], 1) / 80)
            self.B.append(b)

    def save(self):
        s_dt = str(datetime.datetime.now()).split(".")[0]
        d = self.save_path + s_dt
        d = d.replace(' ', '_')
        os.makedirs(d)
        # pickle.dump(self.dims, open(dir + '/dims'))
        for i in range(0, len(self.W)):
            w = self.W[i]
            np.save(d + "/W_" + str(i), w, allow_pickle=False)
        for i in range(0, len(self.B)):
            b = self.B[i]
            np.save(d + "/B_" + str(i), b, allow_pickle=False)
        print('network saved to ' + d)

    def load(self, subdir):
        self.W = []
        self.B = []
        d = self.save_path + subdir
        for i in range(0, self.n_layers - 1):
            w = np.load(d + "/W_" + str(i) + ".npy", allow_pickle=False)
            self.W.append(w)
            b = np.load(d + "/B_" + str(i) + ".npy", allow_pickle=False)
            self.B.append(b)
        print('network loaded from ' + d)

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

    def train(self, training_data, validation_data=None, batch_size=1, n_epoch=1):
        train_size = len(training_data)
        for epoch in xrange(0, n_epoch):
            np.random.shuffle(training_data)
            batches = [training_data[k : k+batch_size]
                       for k in xrange(0, train_size, batch_size)]
            err_sum = 0.0
            for batch in batches:
                nabla_W = None
                nabla_B = None
                for t in batch:
                    x = t[0]
                    y = t[1]
                    self.forward(x)
                    err = mse(y, self.get_outputs())
                    err_sum += err
                    nabla_w, nabla_b = self.backprop(y)

                    if nabla_W is None:
                        nabla_W = nabla_w
                    else:
                        for i in range(0, len(nabla_W)):
                            nabla_W[i] = nabla_W[i] + nabla_w[i]

                    if nabla_B is None:
                        nabla_B = nabla_b
                    else:
                        for i in range(0, len(nabla_B)):
                            nabla_B[i] = nabla_B[i] + nabla_b[i]

                # update weights and biases
                for i in range(0, self.n_layers - 1):
                    self.W[i] = self.W[i] - self.lr * nabla_W[i]
                    self.B[i] = self.B[i] - self.lr * nabla_B[i]

            # print error rate
            print("epoch: %d, error: %f" % (epoch, err_sum / train_size))

            # validation
            if validation_data is not None:
                total_count = len(validation_data)
                count = 0
                for v in validation_data:
                    x = v[0]
                    y = v[1]
                    self.forward(x)
                    y0_v = self.get_outputs()
                    y0 = np.argmax(y0_v)
                    if y0 == y:
                        count += 1

                print("epoch: %d, validation rate: %f" % (epoch, 1.0 * count / total_count))

    def backprop(self, y):
        nabla_w = []
        nabla_b = []
        i = self.n_layers - 2

        # a = self.A[i]
        delta = mse_p(self.A[i], y) * sigmoid_p(self.Z[i])
        nw = np.matmul(delta, self.A[i-1].transpose())
        nabla_w.append(nw)
        nabla_b.append(delta)

        # self.update_weights(i, delta)
        for i in range(self.n_layers - 3, -1, -1):
            delta = np.matmul(self.W[i+1].transpose(), delta) * relu_p(self.Z[i])
            if i == 0:
                a = self.inputs
            else:
                a = self.A[i - 1]
            nw = np.matmul(delta, a.transpose())
            nabla_w.append(nw)
            nabla_b.append(delta)

        nabla_w.reverse()
        nabla_b.reverse()

        return nabla_w, nabla_b

    # def update_weights(self, l, delta):
    #     d = (delta * self.lr)
    #     self.B[l] -= d
    #     if l == 0:
    #         a = self.inputs
    #     else:
    #         a = self.A[l-1]
    #     d = (np.matmul(delta, a.transpose()) * self.lr)
    #     self.W[l] -= d
    #     pass
