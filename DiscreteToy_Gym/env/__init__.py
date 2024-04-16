# -*- coding: utf-8 -*-
# @Time : 2024/4/15 22:01
# @Author : yysgz
# @File : __init__.py.py
# @Project : REINFORCE_tf2.py
# @Description :

from gym.envs.registration import register

register(
    id='Toy-v0',
    entry_point='env.toy_env:ToyEnv',
)
