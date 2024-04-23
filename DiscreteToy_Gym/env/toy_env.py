# -*- coding: utf-8 -*-
# @Time : 2024/4/15 22:01
# @Author : yysgz
# @File : ball_env.py
# @Project : REINFORCE_tf2.py
# @Description :

import gym
from gym import spaces

class ToyEnv(gym.Env):
    def __init__(self):
        """
        为了符合 Gym 的标准，你需要将动作空间（action space）和状态空间（state space）定义为 Gym 中的空间（Space）对象，而不是简单的 Python 列表。
        你可以使用 Gym 提供的空间类来定义这些空间。
        """
        # 动作空间：两个离散动作，['left', 'right']
        self.action_space = spaces.Discrete(2)
        # 状态空间：三个离散状态，分别表示s0、s1、s2
        self.observation_space = spaces.Discrete(3)
        # 状态转移表：定义了每个状态下执行动作后的下一个状态
        self.state_transition = {  # 状态转移表
            0: {0: 0, 1: 1},
            1: {0: 0, 1: 2}
        }
        self.reward = {0: -1, 1:-1, 2:10}  # num_actions * num_states=6 将状态映射为reward
        self.state = 0

    def step(self, action):
        next_state = self.state_transition[self.state][action]  # 通过两个关键字查找状态转移表中的后续状态
        reward = self.reward[next_state]
        done = (next_state == 2)
        info = {}
        print('state: {}, action: {}, reward: {}, next_state: {}'.format(self.state, action, reward, next_state))
        self.state = next_state

        return next_state, reward, done, info

    def reset(self):
        self.state = 0
        return self.state

    def render(self, mode='human'):  # pop up window
        draw = ['-' for i in range(self.observation_space.n)]
        draw[self.state] = 'o'
        draw = ''.join(draw)
        print(draw)
