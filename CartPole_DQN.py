# -*- coding: utf-8 -*-
# @Time : 2024/4/3 11:58
# @Author : yysgz
# @File : CartPole_DQN.py
# @Project : RL_models

import gym
from matplotlib import animation  # 动画
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy as sp

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Net(nn.Module):
    def __init__(self, n_states, n_actions):  # n_states=4; n_actions=2
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 10)  # observation num
        self.fc2 = nn.Linear(10, n_actions)  # action probabilities.
        self.fc1.weight.data.normal_(0, 0.1)  # 对全连接层权重进行初始化操作，(0, 0.1)是一个正态分布函数，其中0是均值，0.1是标准差
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out  # tensor (1, 2)

class DQN:
    def __init__(self, n_states, n_actions):
        print("<DQN init>")
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.n_states = n_states
        self.n_actions = n_actions
        # 使用变量
        self.learn_step_counter = 0  # target网络学习计数
        self.memory_counter = 0  # 记忆计数
        self.memory = np.zeros((2000, 2*n_states +1 +1))  # s, s', a, r
        self.cost = []
        self.done_step_list = []

    def choose_action(self, state, epsilon):  # state: ndarray (4,), [-0.00929577  0.03401177 -0.02519303  0.0058281 ]
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # dim=0, 扩展tensor
        if np.random.uniform() < epsilon:
            action_value = self.eval_net(state)
            action = torch.max(action_value, 1)[1].data.numpy()[0]  # 在dim=1上返回张量的最大值；torch.max返回一个tuple(最大值所在的索引)；data.numpy将tensor张量转换为numpy数组；-》action：1
        else:
            action = np.random.randint(0, self.n_actions)
        # print('actions: ', action)
        return action

    def store_transition(self, state, action, reward, next_state):  # 移动轨迹transition
        print('<store_transitioin>')
        transition = np.hstack((state, [action, reward], next_state))  # 水平方向堆叠，
        index = self.memory_counter % 2000  # 取余运算。满了覆盖旧的
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        print("<learn>")
        # target net 更新频率，用于预测，不会及时更新参数
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 使用记忆库中的批量数据
        sample_index = np.random.choice(2000, 16)  # 2000个中随机抽取16个作为batch_size
        memory = self.memory[sample_index, :]  # 取的记忆单元，并逐个提取
        state = torch.FloatTensor(memory[:, :self.n_states])
        action = torch.LongTensor(memory[:, self.n_states:self.n_states + 1])
        reward = torch.LongTensor(memory[:, self.n_states + 1: self.n_states + 2])
        next_state = torch.FloatTensor(memory[:, self.n_states + 2:])

        # 计算loss, q_eval: 所采取动作的预测value, q_target: 所采取动作的实际value
        q_eval = self.eval_net(state).gather(1, action)  # eval_net->(64,4)->按照action索引提取出q_value
        q_next = self.target_net(next_state).detach()
        q_target = reward + 0.9 * q_next.max(1)[0].unsqueeze(1)  # label
        loss = self.loss(q_eval, q_target)  # td error
        self.cost.append(loss)
        # 反向计算
        self.optimizer.zero_grad()  # 梯度重置
        loss.backward()  # 反向求导
        self.optimizer.step()  # 更新模型参数

    def plot_cost(self):
        plt.subplot(1, 2, 1)
        plt.plot(np.arrange(len(self.cost)), self.cost)
        plt.xlabel('step')
        plt.ylabel('cost')

        plt.subplot(1, 2, 2)
        plt.plot(np.arrange(len(self.done_step_list)), self.done_step_list)
        plt.xlabel('step')
        plt.ylabel('done step')
        plt.show()

def save_gif(frames):
    patch = plt.imshow(frames[0])  # 补丁
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save('./imgs/CartPole_DQN.gif', writer='pillow', fps=30)


if __name__=="__main__":
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    frames = []
    # train
    counter = 0
    done_step = 0
    max_done_step = 0
    num = 200000
    negative_reward = -10.0
    x_bound = 1.0

    state = env.reset()[0]  # tuple:2, ([-0.01098062 -0.00836728 -0.0145362   0.00138723], {})
    model = DQN(n_states=4, n_actions=2)  # states_nmu=4, 说明应该取state[0]
    model.cost.clear()
    model.done_step_list.clear()
    for i in range(num):
        # env.render()
        epsilon = 0.9 + i / num * (0.95-0.9)  # 这是 ε-贪心策略的实现，用于在训练过程中逐渐降低探索的概率。初始时，探索概率为 0.9，随着训练的进行，逐渐线性地减小到 0.95
        action = model.choose_action(state, epsilon)  # 根据给定的状态以及探索概率选择动作
        state_old = state
        # test = env.step(action)  # ([-0.03049995 -0.14930178 -0.0210299   0.24700025], 1.0, False, False, {})
        state, reward, done, info = env.step(action)[0: 4]  # env 是环境对象，step 方法接受动作作为参数，并返回新的状态、奖励、是否终止以及其他信息。
        x, x_dot, theta, theta_dot = state
        # 通过监测倒立摆的位置和角度，计算相应的奖励。如果位置和角度超过了阈值范围，则分别给予负奖励。如果在阈值范围内，则根据偏离阈值的程度给予适当的负奖励。最终将位置和角度对应的奖励相加得到最终的奖励。
        if (abs(x) > x_bound):
            r1 = 0.5 * negative_reward
        else:
            r1 = negative_reward * abs(x) / x_bound + 0.5 * (-negative_reward)  # 4.68559
        # θ点（theta dot）通常表示倒立摆杆的角速度，即倒立摆杆相对于竖直方向的旋转速度。这个角速度的符号可以表示倒立摆杆是向左旋转还是向右旋转，正值表示向右旋转，负值表示向左旋转。
        if (abs(theta) > env.theta_threshold_radians):  # 阈值弧度, theta_threshold_radians=0.20943951023931953
            r2 = 0.5 * negative_reward
        else:
            r2 = negative_reward * abs(theta) / env.theta_threshold_radians + 0.5 * (-negative_reward)

        reward = r1 + r2  # 8.052092992420423
        if done:
            reward += negative_reward

        model.store_transition(state_old, action, reward, state)
        if (i > 2000 and counter % 10 ==0):
            model.learn()
            counter = 0
        counter += 1
        done_step += 1
        if (done):
            if (done_step > max_done_step):
                max_done_step = done_step
            state = env.reset()[0]
            model.done_step_list.append(done_step)
            done_step = 0

    # test
    state = env.reset()[0]
    for _ in range(400):
        frames.append(env.render())
        action = model.choose_action(state, 1.0)
        state, reward, done, info = env.step(action)[0:4]
        if (done):
            state = env.reset()[0]
            print('test try again')
            break
    env.close()
    save_gif(frames)