# -*- coding: utf-8 -*-
# @Time : 2024/4/11 15:08
# @Author : yysgz
# @File : normalized_actions.py
# @Project : REINFORCE_tf2.py
# @Description :

import gym

class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        action = (action + 1) / 2  # [-1,1] -> [0,1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action