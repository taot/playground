import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def show0(data_list, idx):
    data = data_list[0]
    labels = data_list[1]
    # tpl = data_list[index]
    print 'digit: ', unvectorized_result(labels[idx])
    m = np.reshape(data[idx], (28, 28))
    plt.imshow(m)
    plt.show()

def show(t):
    data = t[0]
    label = t[1]
    print 'digit: ', unvectorized_result(label)
    m = np.reshape(data, (28, 28))
    plt.imshow(m)
    plt.show()

def unvectorized_result(vec):
    if np.ndarray == type(vec):
        for x in range(0, len(vec)):
            if vec[x, 0] == 1:
                return x
    else:
        return vec
