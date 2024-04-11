# -*- coding: utf-8 -*-
# @Time : 2024/4/11 16:20
# @Author : yysgz
# @File : reinforce_continuous.py
# @Project : REINFORCE_tf2.py
# @Description : 通过policy network π来控制agent运动

import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable

# CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pi = Variable(torch.FloatTensor([math.pi])).to(device)

def normal(x, mu, sigma_sq):
    a = (-1 * (Variable(x) - mu).pow(2) / (2*sigma_sq).exp())
    b = 1/(2 * sigma_sq * pi.expand_as(sigma_sq).sqrt())
    return a*b

class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        self.num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, self.num_outputs)
        self.linear2_ = nn.Linear(hidden_size, self.num_outputs)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = inputs
        x = self.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)

        return mu, sigma_sq

class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.softplus = nn.Softplus()
        self.model = Policy(hidden_size, num_inputs, action_space).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def select_action(self, state):
        mu, sigma_sq = self.model(Variable(state).to(device))
        sigma_sq = self.softplus(sigma_sq)

        eps = torch.randn(mu.size()).to(device)
        # calculate the probability
        action = (mu + sigma_sq.sqrt() * Variable(eps)).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5 * ((sigma_sq+2*pi.expand_as(sigma_sq)).log() + 1)

        log_prob = prob.log()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss -= ((log_probs[i] * (Variable(R).expand_as(log_probs[i])).to(device)).sum() + (0.0001 * entropies[i].to(device)).sum())
        loss /= len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_((self.model.parameters(), 40))
        self.optimizer.step()
