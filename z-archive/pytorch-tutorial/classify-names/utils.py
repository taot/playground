from __future__ import unicode_literals, print_function, division

import unicodedata
import string
from io import open
import glob
import random

import numpy as np
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def findFiles(path):
    return glob.glob(path)

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def read_all_categories(path):
    category_lines = {}
    all_categories = []
    for filename in findFiles(path + '/*.txt'):
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    return all_categories, category_lines

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor



def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

#
# Data Loader
#
class DataLoader:

    def __init__(self, path):
        all_categories, category_lines = read_all_categories(path)
        self.all_categories = all_categories
        self.category_lines = category_lines
        self.n_categories = len(all_categories)

    def randomTrainingExample(self):
        category = randomChoice(self.all_categories)
        line = randomChoice(self.category_lines[category])
        category_tensor = Variable(torch.LongTensor([self.all_categories.index(category)]))
        line_tensor = Variable(lineToTensor(line))
        return category, line, category_tensor, line_tensor

    def categoryFromOutput(self, output):
        top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
        category_i = top_i[0][0]
        return self.all_categories[category_i], category_i
    