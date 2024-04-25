import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper();

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

imgplot = plt.imshow(training_data[0][0])
plt.show()
