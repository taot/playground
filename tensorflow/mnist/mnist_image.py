import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# reload a module in python3
#
# import imp
# imp.reload(mnist_image)


def show(dataset, index):
    img = dataset.images[index]
    print('digit: %d' % unvectorized_result(dataset.labels[index]))
    m = np.reshape(dataset.images[index], (28, 28))
    plt.imshow(m)
    plt.show()


def unvectorized_result(vec):
    print(vec)
    if np.ndarray == type(vec):
        for x in range(0, len(vec)):
            if vec[x] == 1:
                return x
    else:
        return vec
