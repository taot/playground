import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import gym
from collections import namedtuple
import random
import math

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        t = Transition(*args)
        self.memory[self.position] = t
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


env = gym.make('CartPole-v0')

BATCH_SIZE = 32
memory = ReplayMemory(10000)
gamma = 0.95
eps_start = 0.9
eps_end = 0.05
EPS_DECAY = 1000

model = DNN()
if use_cuda:
    model = model.cuda()
optimizer = optim.RMSprop(model.parameters())

episode = 0
steps_done = 0


def select_action(state):
    global steps_done
    eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() < eps:
        return LongTensor([[random.randrange(2)]])
    # state = torch.from_numpy(state).type(Tensor)
    inputs = Variable(state.view(1, -1))
    outputs = model(inputs)
    a = torch.max(outputs.data, 1)[1].view(1, 1)
    return a


def optimize():
    if len(memory) < BATCH_SIZE:
        return -1

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    non_final_mask = ByteTensor(list(map(lambda x: x is not None, batch.next_state)))
    next_state_batch = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
    reward_batch = Variable(torch.cat(batch.reward))

    state_action_values = model(state_batch).gather(1, action_batch)
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = torch.max(model(next_state_batch), 1)[0]
    next_state_values.volatile = False
    expected_state_action_values = next_state_values * gamma + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    # print("loss: %f" % loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data[0]


def simulate():
    done = False
    state = env.reset()
    state = Tensor(state).view(1, -1)
    loss_sum = 0
    loss_count = 0
    duration = 0
    while not done:
        action = select_action(state)
        next_state, reward, done, _ = env.step(action[0, 0])
        env.render()
        state = Tensor(state).view(1, -1)
        if done:
            next_state = None
        else:
            next_state = Tensor(state).view(1, -1)
        reward = Tensor([reward])
        memory.push(state, action, next_state, reward)
        loss = optimize()
        duration += 1
        if loss > 0:
            loss_sum += loss
            loss_count += 1
    avg_loss = -1
    if loss_count > 0:
        avg_loss = loss_sum / loss_count
    print("duration %d, average loss: %f" % (duration, avg_loss))


for i in range(50):
    for j in range(30):
        print("Episode %d, Turn %d" % (episode, j))
        simulate()
    episode += 1
