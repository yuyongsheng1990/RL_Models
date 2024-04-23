# -*- coding: utf-8 -*-
# @Time : 2024/4/16 16:53
# @Author : yysgz
# @File : ball_env.py
# @Project : REINFORCE_tf2.py
# @Description :

import gym
from gym import spaces

import math
import time
from ContinuousToy_Gym.env import rendering

import numpy as np

# VIEWPORT_W, VIEWPORT_H为地图宽高
Viewport_W = 200
Viewport_H = 200

gamma = 0.2
max_ball_num = 15
max_ball_score = 200

Ball_blue = 0
Ball_red = 1

Ball_start_id = 0

def GeneratedBallID():
    global Ball_start_id

    Ball_start_id += 1

    return Ball_start_id

def CheckBound(low, high, value):
    if value > high:
        value -= (high - low)
    elif value < low:
        value += (high - low)
    return value

class Ball():
    def __init__(self, x: np.float32, y: np.float32, score: np.float32, angle: int, t: int):
        '''
            :param x: coordinate
            :param y: coordinate
            :param score: score of ball
            :param way: move direction of ball, in radians 弧度
            :param t: type of ball, self or other
        '''
        self.x = x
        self.y = y
        self.s = CheckBound(0, max_ball_score, score)
        self.r = angle * 2 * math.pi / 360.0  # angle to radians，角度转换为弧度
        self.t = t  # type

        self.id = GeneratedBallID()  # ball id
        self.lastupdate = time.time()  # last update time, used to calculate ball move
        self.timescale = 100  # time scale, used to calculate ball move

    def update(self, angle):  # update ball, include position
        # can only change self way
        if self.t == Ball_red:
            self.r = angle * 2 * math.pi / 360.0  # angle to radians

        speed = 10.0 / self.s  # score to speed
        now = time.time()
        self.x += math.cos(self.r) * speed * (now-self.lastupdate) * self.timescale  # direction * speed * time = distance
        self.y += math.sin(self.r) * speed * (now-self.lastupdate) * self.timescale  # direction * speed * time = distance

        self.x = CheckBound(0, Viewport_W, self.x)
        self.y = CheckBound(0, Viewport_H, self.y)

        self.lastupdate = now  # update time

    def addscore(self, score: np.float32):
        self.s += score

    def minusscore(self, score: np.float32):
        self.s -= score

    def state(self):
        return [self.x, self.y, self.s, self.t]

class BallEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # self.seed()
        self.viewer = None  # render viewer
        self.scale = 3  # viewer size scale

        self.action_space = spaces.Discrete(36)
        ''' 
            [[x, y, score, type],
             [x, y, score, type],
              ...                ]
        '''
        # 使用low、high创一个shape的Box空间，其中每个维度的取值范围是[low, high].
        self.observation_space = spaces.Box(low=0, high=Viewport_H, shape=(max_ball_num*4, ), dtype=np.float32)  # Box(0,200,(60,),float32)

        self.balls = []
        self.state = np.zeros((max_ball_num * 4), dtype=np.float32)  # (60,)
        self.reset()

    def reset(self):
        self.balls = []
        # random generate blue balls
        min = max_ball_score  # 200
        max = 0
        for i in range(max_ball_num - 1):
            tmp = self.randball(Ball_blue)

            if tmp.s < min:
                min = tmp.s

            if tmp.s > max:
                max = tmp.s

            self.balls.append(tmp)

        # random generate red ball
        self.redball = self.randball(Ball_red, (min+max)/2)

        # add to ball list
        self.balls.append(self.redball)

        # update state
        self.state = np.vstack([ball.state() for ball in self.balls])  # vertical，用于垂直堆叠数组  (15, 4)

        return self.state.reshape(max_ball_num * 4, )  # (60, )

    def step(self, action: int):
        reward = 0.0
        done = False

        action = 10 * action  # input angle to ball.

        # update ball
        for ball in self.balls:
            ball.update(action)

        '''
            Calculate Ball Eat
            if ball A contains ball B's center, and A's score > B's score, A eats B.
        '''
        _new_ball_types = []
        for _, A_ball in enumerate(self.balls):
            for _, B_ball in enumerate(self.balls):

                if A_ball.id == B_ball.id:
                    continue

                # radius of ball A
                A_radius = math.sqrt(A_ball.s / math.pi)  # 球面积s = πr2,

                # vector AB
                AB_x = math.fabs(A_ball.x - B_ball.x)  # 绝对值
                AB_y = math.fabs(A_ball.y - B_ball.y)

                # B is out of A
                if AB_x > A_radius or AB_y > A_radius:
                    continue

                # B is out of A
                if math.sqrt(AB_x * AB_x + AB_y * AB_y) > A_radius:
                    continue

                # otherwise, blue ball would be eaten, negative reward
                if B_ball.t == Ball_red:
                    reward -= B_ball.s
                    done = True
                    # delete A
                    _new_ball_types.append(A_ball.t)
                    self.balls.remove(A_ball)

                # A eat B
                A_ball.addscore(B_ball.s)

                # calculate reward
                if A_ball.t == Ball_red:
                    reward += B_ball.s
                    A_ball.addscore(gamma * B_ball.s)

                # delete B
                _new_ball_types.append(B_ball.t)
                self.balls.remove(B_ball)

        # generate new balls to max_ball_num
        for _, val in enumerate(_new_ball_types):
            self.balls.append(self.randball(int(val)))

        self.state = np.vstack([ball.state() for ball in self.balls])  # (15,4)

        return self.state.reshape(max_ball_num * 4,), reward, done, {}

    def render(self, mode='human'):
        # create viewer
        if self.viewer is None:
            self.viewer = rendering.Viewer(Viewport_W * self.scale, Viewport_H * self.scale)

        # add ball to viewer
        for item in self.state:
            _x, _y, _s, _t = item[0] * self.scale, item[1] * self.scale, item[2], item[3]  # _x, _y, _s, _angle, _t

            transform = rendering.Transform()
            transform.set_translation(_x, _y)

            # add a circle
            # center: (x, y)
            # radius: sqrt(score/pi)
            # colors: self in red, other in blue
            self.viewer.draw_circle(math.sqrt(_s / math.pi) * self.scale, 30, color=(_t, 0, 1)).add_attr(transform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        return

    @staticmethod
    def randball(_t: int, _s: float=0):  # type _t; score _s
        if _s <= 0:
            _s = np.random.rand(1)[0] * max_ball_score  # 生成一个长度为1的在[0,1)区间的随机数; [0]取第一个元素
        _x, _y = np.random.rand(1)[0] * Viewport_W, np.random.rand(1)[0] * Viewport_H  # ball coordinates
        _angle = int(np.random.rand(1)[0]*360)
        _b = Ball(_x, _y, _s, _angle, _t)  # 随机生成球
        return _b

# if __name__ == '__main__':
#     env = BallEnv()
#
#     num_steps = 1000
#     i = 0
#     while i < num_steps:
#         env.step(15)
#         env.render()
#         i += 1