# -*- coding: utf-8 -*-
# @Time : 2024/4/8 9:13
# @Author : yysgz
# @File : agents.py
# @Project : RL_models
# @Description : Q-Learning是一种基于值函数的强化学习算法，用于学习在给定环境下采取行动的最优策略。
"""
1. 初始化Q-table：创建一个Q-table，其行表示state，列表示action。初始时，Q-table可以使零矩阵，也可以是随机初始化的值。
2. choose action：根据当前state，根据一定的策略(i.e., ε-greedy策略)，选择一个action。
    - ε-greedy策略允许agent以一定的概率ε选择随机动作，以便探索环境。
3. 执行action：在env中执行action，获得next state和reward。
4. 更新Q-table：使用Q-Learning更新规则更新Q-table，
    - 更新规则：Q(s,a) <- Q(s,a) + α * [r + γ * max_a Q(s',a') - Q(s,a)]
    @ Q(s,a)是状态s下执行动作a的q-value。
    @ r是状态s执行动作a后的奖励
    @ s'是执行动作a后的新状态。
    @ α是学习率，控制着q-value的更新速度。
    @ γ是折扣因子discount factor，表示agent对未来奖励的重视程度。
5. 终止条件：重复step2-4，直到达到终止条件，或最大迭代次数。
6. 策略提取，根据学习到的Q-table，选择最后策略。通常选择每个状态下具有最大q-value的action。
"""

import abc
from typing import Optional  # Optional类型表示一个值可以使某种类型的实例，也可以是None

import numpy as np

import sys
sys.path.append(r'C:\Users\yysgz\OneDrive - Macquarie University\Desktop\RL_models\CartPole_Q_Learning\cartpole')

from entities import Action, Observation, Reward, State

class Agent(abc.ABC):
    @abc.abstractmethod
    def begin_episode(self, observation: Observation) -> Action:
        pass

    @abc.abstractmethod
    def act(self, observation: Observation, reward: Reward) -> Action:
        pass

class RandomActionAgent(Agent):
    'Agent that has no learning behavior and acts randomly at all times.'
    def __init__(self, random_state: np.random.RandomState = None):
        self.random_state = random_state or np.random

    def begin_episode(self, observation: Observation) -> Action:  # 返回action
        return self.random_state.choice([0, 1])  # type: ignore

    def act(self, observation: Observation, reward: Reward) -> Action:
        return self.random_state.choice([0, 1])  # type: ignore

class QLearningAgent(Agent):
    'Agent that learns from experience using tabular Q-Learning.'
    def __init__(self, learning_rate: float = 0.2, discount_factor: float = 1.0, exploration_rate: float = 0.5,  # 折扣率、探索率、探索衰减率
                 exploration_decay_rate: float = 0.99, random_state: np.random.RandomState = None):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.random_state = random_state or np.random

        self.state: Optional[State] = None
        self.action: Optional[Action] = None

        # Discretize the continous state space for each of the 4 features.
        num_discretization_bins = 7
        self._state_bins =[
            # np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIGITIZED))
            # Cart position.
            self._discretize_range(-2.4, 2.4, num_discretization_bins),
            # Cart velocity. 速度
            self._discretize_range(-3.0, 3.0, num_discretization_bins),
            # Pole angle.
            self._discretize_range(-0.5, 0.5, num_discretization_bins),
            # Pole velocity.
            self._discretize_range(-2.0, 2.0, num_discretization_bins),
        ]

        # Create a clearn Q-Table, where each state is a row and each action is a column.
        self._max_bins = max(len(bin) for bin in self._state_bins)  # 6
        self._num_states = (self._max_bins + 1) ** len(self._state_bins)  # 幂运算，7^4
        self._num_actions = 2
        self._q = np.zeros(shape=(self._num_states, self._num_actions))  # 创建q-table，根据shape返回全是0的数组

    @staticmethod  # 离散化方法，将连续的观测空间离散化为状态空间中的索引值。
    def _discretize_range(lower_bound: float, upper_bound: float, num_bins: int) -> np.ndarray:
        return np.linspace(lower_bound, upper_bound, num_bins + 1)[1: -1]  # linspace创建等间隔的一维数组，等差数列

    @staticmethod
    def _discretize_value(value: float, bins: np.ndarray) -> np.ndarray:
        return np.digitize(x=value, bins=bins)  # 返回一个与x大小相同的数组，表示每个元素所属bins的索引。其中x是要分箱的值，bins是一维数组，表示分箱的边界。

    def _build_state_from_observation(self, observation: Observation) -> State:
        # Discretize the observation features and reduce them to a single integer.
        # The resulting integer value will correspond to the row number in the Q-Table.
        state = sum(
            self._discretize_value(feature, self._state_bins[i]) * ((self._max_bins + 1) ** i)
            for i, feature in enumerate(observation)
        )
        return state  # type: ignore

    def begin_episode(self, observation: Observation) -> Action:  # 在每个episode开始时，根据当前观测值选择action
        # Reduce exploration over time
        self.exploration_rate *= self.exploration_decay_rate

        # Get the action for the initial state.
        self.state = self._build_state_from_observation(observation)

        return np.argmax(self._q[self.state])  # state row from self._q

    def act(self, observation: Observation, reward: Reward) -> Action:
        # 更新Q-table。这里传入的observation和reward是指next state、reward，然后以一定概率更换q-table中对应的best action。
        # self.state和self.action其实对应着当前state和action
        next_state = self._build_state_from_observation(observation)  # state int value

        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = (1 - self.exploration_rate) <= self.random_state.uniform(0, 1)  # bool comparison: 1-探索率=利用率；(0,1)内均匀分布的随机数。
        if enable_exploration:
            next_action = self.random_state.randint(0, self._num_actions)  # 生成一个位于(0, self._num_actions)之间的整数
        else:
            next_action = np.argmax(self._q[next_state])

        # Learn: update Q-Table based on current reward and future action.
        '''更新规则：Q(s,a) <- Q(s,a) + α * [r + γ * max_a Q(s',a') - Q(s,a)]'''
        self._q[self.state, self.action] += self.learning_rate * (reward +
                                                                  self.discount_factor * max(self._q[next_state, :]) -
                                                                  self._q[self.state, self.action])

        self.state = next_state
        self.action = next_action
        return next_action  # type: ignore