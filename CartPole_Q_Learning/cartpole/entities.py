# -*- coding: utf-8 -*-
# @Time : 2024/4/8 9:15
# @Author : yysgz
# @File : entities.py
# @Project : RL_models
# @Description :

import dataclasses
from typing import List

import numpy as np
import pandas as pd

Action = int
State = int
Observation = np.ndarray
Reward = float

@dataclasses.dataclass
class EpisodeHistoryRecord:  # 这是一个数据类，存储每个episode的历史记录。
    episode_index: int
    episode_length: int
    is_successful: bool

class EpisodeHistory:  # 这是一个用于存储episode回合的历史记录，并检查是否达到目标的类。
    'Stores the history of episode durations and checks if the goal has been achieved'
    def __init__(self, max_timesteps_per_episode: int = 200, goal_avg_episode_length: int = 195,
                 goal_consecutive_episodes: int = 100,) -> None:  # 连续的, consecutive
        self._records: List[EpisodeHistoryRecord] = []
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.goal_avg_episode_length = goal_avg_episode_length
        self.goal_consecutive_episodes = goal_consecutive_episodes

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, episode_index: int) -> EpisodeHistoryRecord:
        return self._records[episode_index]

    def all_records(self) -> List[EpisodeHistoryRecord]:
        return list(self._records[:])

    @property
    def last_episode_index(self) -> int:
        return len(self._records) - 1

    def record_episode(self, data: EpisodeHistoryRecord) -> None:
        self._records.append(data)

    def most_recent_lengths(self, n: int) -> np.ndarray:  # 最近几个episode(回合)的长度
        recent_records = self._records[max(0, len(self) - n): len(self)]  # self指向一个类实例的引用，通过self，方法可以访问该类成员变量_records。
        return np.array([rec.episode_length for rec in recent_records])

    def most_recent_rolling_mean_lengths(self, n: int, window_size: int=101) -> np.ndarray:
        recent_lengths = self.most_recent_lengths(n + window_size)
        # pd.Series.rolling方法计算滚动窗口的means，rolling方法创建一个滚动窗口对象；mean对窗口中的值求均值操作。
        rolling_means = pd.Series(recent_lengths).rolling(window=window_size, min_periods=0).mean()  # 滚动窗口的均值
        return rolling_means.values[-n: ]   # 取出-n个值

    def is_goal_reached(self) -> bool:  # 用于检查是否达到了目标，达到目标的条件是最近连续一定数量的episode的平均长度是否超过了预设目标平均episode长度。
        recent_lengths = self.most_recent_lengths(self.goal_consecutive_episodes)
        avg_length = np.average(recent_lengths) if len(recent_lengths) > 0 else 0
        return bool(avg_length >= self.goal_avg_episode_length)

