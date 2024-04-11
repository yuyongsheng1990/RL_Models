# -*- coding: utf-8 -*-
# @Time : 2024/4/11 17:16
# @Author : yysgz
# @File : reinforce_discrete.py
# @Project : REINFORCE_tf2.py
# @Description : 通过policy network π来控制agent运动

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
import pdb

# CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Policy(nn.Module):
    def __init__(self, num_inputs, hidden_size, action_space):
        super(Policy, self).__init__()
        num_outputs = action_space.n

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.relu(x)
        action_scores = self.linear2(x)

        return self.softmax(action_scores)

class REINFORCE:
    def __init__(self, num_inputs, hidden_size, action_space):  # 128; num_inputs=4; Discrete(2)
        self.model = Policy(num_inputs, hidden_size, action_space).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()  # 将model设置为train模式，可以BP更新梯度。

    def select_action(self, state):
        probs = self.model(Variable(state).to(device))  # action probability, tensor: (1,2)
        # torch.multinomial作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的"下标"，采样数目num_samples=1, input张量可以看成一个权重张量，元素代表被采样权重，replacement表示是否有放回取样。
        action = probs.multinomial(num_samples=1).data  # action, tensor:(1,1)
        prob = probs[:, action[0,0]].view(1,-1)
        log_prob = prob.log()  # 取log
        entropy = -(probs * probs.log()).sum()

        return action[0], log_prob, entropy

    def update_parameters(self, log_probs, rewards, entropies, gamma):
        R = torch.zeros(1,1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]  # discount factor, Ut = Rt + gamma * Ut+1
            # 离散动作：∑a ▽π(a|s;θ)/▽θ * Qπ(s,a); REINFORCE Algorithm, 令ut = qt。
            # 即Vπ的导数=梯度，变大！
            # policy gradient ascent, 反映到model实现上就是loss相减，因为torch默认是梯度下降更新。
            loss = loss - (log_probs[i] * (Variable(R).expand_as(log_probs[i])).to(device)).sum() - (0.0001 * entropies[i].to(device)).sum()
        loss /= len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()



