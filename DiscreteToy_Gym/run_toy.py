# -*- coding: utf-8 -*-
# @Time : 2024/4/15 22:51
# @Author : yysgz
# @File : run_toy.py
# @Project : RL_models
# @Description :

import gym
import random
import time

from env.toy_env import ToyEnv  # 导入env路径

# shut down warning info
import warnings
warnings.filterwarnings("ignore")

env = gym.make('Toy-v0')
state = env.reset()

while True:
    action = random.randint(0, env.action_space.n-1)  # 随机动作
    next_state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.5)
    if done: break