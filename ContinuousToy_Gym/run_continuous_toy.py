# -*- coding: utf-8 -*-
# @Time : 2024/4/16 16:53
# @Author : yysgz
# @File : run_continuous_toy.py
# @Project : REINFORCE_tf2.py
# @Description :

import gym
from env.ball_env import BallEnv  # 导入env路径

# shut down warning info
import warnings
warnings.filterwarnings("ignore")

env = gym.make('Ball-v0')
state = env.reset()

num_episodes = 10000
i = 0
while i < num_episodes:
    env.step(15)
    env.render()
    i += 1