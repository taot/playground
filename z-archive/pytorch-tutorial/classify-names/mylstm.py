import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import utils


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        # self.W_f = Variable(torch.rand(hidden_size, concat_size))
        # self.b_f = Variable(torch.rand(hidden_size, 1))
        # self.W_i = Variable(torch.rand(hidden_size, concat_size))
        # self.b_i = Variable(torch.rand(hidden_size, 1))
        # self.W_c = Variable(torch.rand(hidden_size, concat_size))
        # self.b_c = Variable(torch.rand(hidden_size, 1))
        # self.W_o = Variable(torch.rand(hidden_size, concat_size))
        # self.b_o = Variable(torch.rand(hidden_size, 1))

        self.linear_f = nn.Linear(concat_size, hidden_size)
        self.act_f = nn.Sigmoid()
        self.linear_i = nn.Linear(concat_size, hidden_size)
        self.act_i = nn.Sigmoid()
        self.linear_C_ = nn.Linear(concat_size, hidden_size)
        self.act_C_ = nn.Tanh()
        self.linear_o = nn.Linear(concat_size, hidden_size)
        self.act_o = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        # f = torch.sigmoid(torch.matmul(self.W_f, combined) + self.b_f)
        # i = torch.sigmoid(torch.matmul(self.W_i, combined) + self.b_i)
        # C_ = torch.tanh(torch.matmul(self.W_c, combined) + self.b_c)
        # cell_ = f * cell + i * C_
        # o = torch.sigmoid(torch.matmul(self.W_o, combined) + self.b_o)
        # hidden_ = o * torch.tanh(cell_)
        # return hidden_, cell_

        f = self.act_f(self.linear_f(combined))
        i = self.act_i(self.linear_i(combined))
        C_ = self.act_C_(self.linear_C_(combined))
        cell_ = f * cell + i * C_
        o = self.act_o(self.linear_o(combined))
        hidden_ = o * torch.tanh(cell_)
        output = self.softmax(hidden_)

        return output, hidden_, cell_

    def initHidden(self):
        hidden = Variable(torch.zeros(1, self.hidden_size))
        cell = Variable(torch.zeros(1, self.hidden_size))
        return hidden, cell

def train(net, category_tensor, line_tensor, criterion, learning_rate):
    hidden, cell = net.initHidden()
    net.zero_grad()

    for i in range(line_tensor.size()[0]):
        input = line_tensor[i]
        output, hidden, cell = net.forward(input, hidden, cell)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in net.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

def evaluate(net, line_tensor):
    hidden, cell = net.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden, cell = net.forward(line_tensor[i], hidden, cell)

    return output

def predict(net, loader, input_line, n_predictions=3):
    print('\n> %s' % input_line)
    output = evaluate(net, Variable(utils.lineToTensor(input_line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, loader.all_categories[category_index]))
        predictions.append([value, loader.all_categories[category_index]])

def show_confusion_matrix(net, loader):
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(loader.n_categories, loader.n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = loader.randomTrainingExample()
        output = evaluate(net, line_tensor)
        guess, guess_i = loader.categoryFromOutput(output)
        category_i = loader.all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(loader.n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + loader.all_categories, rotation=90)
    ax.set_yticklabels([''] + loader.all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()
