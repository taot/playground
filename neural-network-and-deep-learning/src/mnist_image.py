import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def show(data_list, index):
    tpl = data_list[index]
    print 'digit: ', unvectorized_result(tpl[1])
    m = np.reshape(tpl[0], (28, 28))
    plt.imshow(m)
    plt.show()

def unvectorized_result(vec):
    if np.ndarray == type(vec):
        for x in range(0, len(vec)):
            if vec[x, 0] == 1:
                return x
    else:
        return vec
