# -*- coding: utf-8 -*-
# @Time : 2024/1/9 9:02
# @Author : yysgz
# @File : CartPole_REINFORCE.py
# @Project : RL_models. 用torch实现OpenAI Gym的CartPole环境，任务是让杆子在小车上保持平衡。
"""
Policy-based function -》policy network π(a|s;θ)
- 动作价值函数 Qπ(st, at) = E(Ut)，评价当前状态st下动作at的好坏，期望消掉了随机变量Ai和Si。
- 状态价值函数 Vπ(st; θ) = ∑a π(a|s;θ) * Qπ(st,a)，评价当前状态下策略函数π的好坏，期望消掉了动作A。
Then, how to make policy function better and better? policy gradient ascent to improve Vπ function
- 关于S对Vπ求期望，消掉S，得到目标函数J(θ)，就只剩下优化network parameters θ的目标函数。
- θ <- θ + β * ▽J(θ)/▽θ ≈ θ + β * ▽V(s;θ)/▽θ，随机梯度，随机性来源于s
- ▽V(s;θ)/▽θ with monte carlo approximation
    - 离散动作：∑a ▽π(a|s;θ)/▽θ * Qπ(s,a)
    - 连续动作：E_A[▽log(π(A|s;θ)/▽θ) * Qπ(s,A)]
==> estimate Qπ(st,at) ≈ qt
Then how to approximate qt?
- REINFORCE Algorithm, 令ut = qt。
- Action-Critic Algorithm，用neural network去估计qt。

REINFORCE Algorithm (REward Increment增量 = nonnegative Factor x Offset Reinforcement偏移强化 x Characteristic Eligibility特征资格)
1. 初始化：初始化policy参数，如神经网络权重。
2. 采样轨迹trajectory：使用当前策略在env中采样的多个轨迹或episodes，记录每个轨迹trajectory中的state、action、reward。
3. 计算return：对于每条轨迹，回报discounted return通常是每一步reward的累积。
4. 计算梯度 gradience：根据每条轨迹中的state、action和return，计算policy关于参数的梯度。这通常通过链式法则来计算策略梯度。
5. 更新参数：使用计算得到的策略梯度来更新策略的参数。通过采用梯度上升来最大化期望回报(未来可能回报的加权和)。
6. 重复step 2-5，直到满足停止条件(如达到最大迭代次数或收敛到一定程度)。
"""

import argparse, math

import numpy as np
import pandas as pd

import gym
from gym import wrappers

import torch
from torch.autograd import Variable
import torch.nn.utils as utils

from normalized_actions import NormalizedActions

from matplotlib import pyplot as plt
'''
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def args_register():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE Example')  # 创建参数对象
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default:0.99)')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N', help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=123, metavar='N', help='random seed (default: 123')
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N', help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=200, metavar='N', help='number of episodes (default: 2000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N', help='number of episodes (default: 128')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--ckpt_freq', type=int, default=100, help='model saving frequency')
    parser.add_argument('--display', type=bool, default=True, help='display or not')
    parser.add_argument('--mean_episodes', default=20, help='the scale of x-axis for plots')

    args = parser.parse_args(args=[])  # 解析参数

    return args

def runAgent(args):
    env_name = args.env_name
    env = gym.make(env_name)
    if type(env.action_space) != gym.spaces.discrete.Discrete:
        from reinforce_continuous import REINFORCE
        env = NormalizedActions(env)
    else:
        from reinforce_discrete import REINFORCE

    episode_records = []
    mean_episode_records = []

    # env.seed(args.seed)   # 高版本不适用
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # print(env.observation_space.shape, env.action_space)

    agent = REINFORCE(env.observation_space.shape[0], args.hidden_size, env.action_space)  # observation_shape=(4,); hid_size=128; Discrete(2)

    dir = 'ckpt_' + env_name
    if not os.path.exists(dir):
        os.mkdir(dir)

    for i_episode in range(args.num_episodes):  # 2000
        state = torch.Tensor([env.reset()[0]])  # initial state
        entropies = []
        log_probs = []
        rewards = []
        for t in range(args.num_steps): # 1000
            action, log_prob, entropy = agent.select_action(state)
            action = action.to(device)  # tensor: (1,)
            # print(env.step((action.numpy()[0])))
            next_state, reward, done, _ = env.step(action.numpy()[0])[0:4]

            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = torch.Tensor([next_state])

            if done or (t == args.num_steps-1):
                episode_records.append(t)
                break

        agent.update_parameters(log_probs, rewards, entropies, args.gamma)  # discount factor

        if i_episode % args.ckpt_freq == 0:
            torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-' + str(i_episode) + '.pkl'))

        start_point = max(0, i_episode - args.mean_episodes)
        mean_episode = np.array(episode_records[start_point:]).mean()
        mean_episode_records.append(mean_episode)

        print("Episode: {}, mean_steps: {}".format(i_episode, mean_episode))

    env.close()
    return mean_episode_records

if __name__=='__main__':
    # define args
    args = args_register()

    # run agent with REINFORCE Algorithm
    mean_episode_records = runAgent(args)

    # plotting
    plt.figure(figsize=(15, 8))
    plt.plot(list(range(len(mean_episode_records))), mean_episode_records, label='Mean_Steps')  # 创建空白折线图，，将其赋值给point_plot
    plt.savefig("./CartPole_REINFORCE.jpg", dpi=300)
    plt.show()