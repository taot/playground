import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt


def display_image(image, label):
    plt.imshow(image, cmap="gray")
    plt.title("%i" % label)
    plt.axis("off")
    plt.show()


def display_images(images, labels, rows, cols, output_labels=None):
    figure = plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(len(images)):
        figure.add_subplot(rows, cols, i+1)
        plt.axis("off")
        if output_labels is not None:
            plt.title("c %i, o %i" % (labels[i], output_labels[i]))
        else:
            plt.title("%i" % labels[i])
        plt.imshow(images[i].squeeze(), cmap="gray")
    plt.show()


class Network1(nn.Module):

    def __init__(self, sizes):
        super(Network1, self).__init__()
        self.sizes = sizes
        layers = [
            nn.Sequential(
                nn.Linear(x, y),
                nn.Sigmoid(),
            )
            for x, y in zip(sizes[:-1], sizes[1:])
        ]
        for i in range(len(layers)):
            setattr(self, "layer{0}".format(i), layers[i])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i in range(len(self.sizes) - 1):
            layer = getattr(self, "layer{0}".format(i))
            x = layer(x)
        return x

    def evaluate(self, test_data_loader):
        self.eval()
        with torch.no_grad():
            total_correct = 0
            total_count = 0
            for images, labels in test_data_loader:
                test_output = self(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                correct = (pred_y == labels).sum().item()
                count = len(labels)
                total_correct += correct
                total_count += count
        return total_count, total_correct

    def SGD(self, training_data, epochs, mini_batch_size, lr, loss_fn, test_data=None):
        training_data_loader = DataLoader(training_data, batch_size=mini_batch_size, shuffle=True, num_workers=0)
        test_data_loader = None
        if test_data is not None:
            test_data_loader = DataLoader(test_data, batch_size=mini_batch_size, shuffle=True, num_workers=0)
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            total_count = 0

            self.train()

            for i, (images, labels) in enumerate(training_data_loader):
                b_x = Variable(images)
                b_y = Variable(labels)
                b_y_encoded = F.one_hot(b_y, 10).float()
                output = self(b_x)
                pred_y = torch.max(output, 1)[1].data.squeeze()

                correct = (pred_y == labels).sum().item()
                count = len(labels)
                total_correct += correct
                total_count += count

                loss = loss_fn(output, b_y_encoded)
                # print(loss)
                # self.loss = loss
                total_loss += (loss.item() * count)

                # clear gradients for this training step
                optimizer.zero_grad()
                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()

            s = "Epoch [{}/{}], loss: {:.5f}, training_accuracy: {:.5f}".format(epoch + 1, epochs, total_loss / total_count, total_correct / total_count)
            if test_data_loader is not None:
                eval_count, eval_correct = self.evaluate(test_data_loader)
                s += ", evaluation {} / {}, evaluation accuracy: {:.5f}".format(eval_correct, eval_count, eval_correct / eval_count)
            print(s)

