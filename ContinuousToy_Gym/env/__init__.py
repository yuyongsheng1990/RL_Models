# -*- coding: utf-8 -*-
# @Time : 2024/4/18 17:18
# @Author : yysgz
# @File : __init__.py.py
# @Project : REINFORCE_tf2.py
# @Description :


from gym.envs.registration import register

register(
    id='Ball-v0',
    entry_point='env.ball_env:BallEnv',
)
