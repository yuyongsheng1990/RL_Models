# -*- coding: utf-8 -*-
# @Time : 2024/1/9 9:02
# @Author : yysgz
# @File : REINFORCE_torch.py
# @Project : RL_models. 用torch实现OpenAI Gym的CartPole环境，任务是让杆子在小车上保持平衡。

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

import torch
import torch.nn as nn
import torch.optim as optim

import gym

# 定义policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# 定义REINFORCE算法
def reinforce(policy, optimizer, states, actions, rewards):
    policy_loss = []
    returns =[]

    # 计算每个time step的return
    for r in rewards:
        R=0
        returns.insert(0, R)
        for t in reversed(r):  # 返回一个反转的迭代器
            R = t + 0.99 * R  # 折扣因子为0.99
            returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 归一化

    # 计算损失函数
    for log_prob, R in zip(policy(states), returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

# training model
def train():
    env = gym.make('CartPole-v1')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    policy = PolicyNetwork(input_size, 128, output_size)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    num_episodes = 1000

    for episode in range(num_episodes):
        # env.reset函数用于重置环境，该函数将使得环境的initial observation重置。
        state = env.reset()  # tuple:2, ([0.02465712 0.04523329 0.0123908  0.03902304], {})
        done = False
        states, actions, rewards = [], [], []

        while not done:
            state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0)
            action_prob = policy(state)
            action = torch.multinomial(action_prob, 1).item()  # 采样函数，根据给定的采样概率对数组进行多次采样，返回采样后的元素下标

            temp = env.step(action)  # step会返回四个值：observation(object), reward(float), done(boolean), info(dict), 其中done表示是否应该reset环境。
            next_state, reward, done, _ = temp[:4]

            states.append(state)
            actions.append(torch.tensor(action, dtype=torch.float32))
            rewards.append(reward)

            state = next_state

        reinforce(policy, optimizer, torch.cat(states), torch.cat(actions), rewards)

        if episode % 10 ==0:
            print(f'Episode {episode}, Total Reward: {sum(rewards)}')

    env.close()

if __name__ == "__main__":
    train()

