# -*- coding: utf-8 -*-
# @Time : 2024/1/8 11:37
# @Author : yysgz
# @File : CartPole.py
# @Project : RL_models

import gym
from matplotlib import animation  # 动画片
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode='rgb_array')
obs = env.reset()

kp = 0.000
kv = -0.002
ka = -0.3
kav = -0.01
ks = -0.000
sum_angle = 0.000
frames = []

def save_gif(frames):
    patch = plt.imshow(frames[0])  # 补丁
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save('./imgs/CartPole_game.gif', writer='pillow', fps=50)

def CalAction(obs):
    action = 0  # o means left, 1 means rights
    global sum_angle
    sum = kp * obs[0][0] + kv * obs[0][1] + ka * obs[0][2] + kav * obs[0][3] + ks * sum_angle
    sum_angle += obs[0][2]
    if sum < 0.0:
        action = 1
    else:
        action = 0
    return action

for step in range(10000):
    frames.append(env.render())  # env.render()  # 弹出窗口, a window pops up
    action = CalAction(obs)
    print('step = %d' % step, 'action = %d' % action)
    obs = env.step(action)
    print("observation: {}, reward: {}, done: {}, info: {}".format(obs[0][0], obs[0][1], obs[0][2], obs[0][3]))


env.close()
save_gif(frames)
