# -*- coding: utf-8 -*-
# @Time : 2024/1/8 17:35
# @Author : yysgz
# @File : REINFORCE_tf2.py
# @Project : RL_models. 用tensorflowOpenAI Gym的CartPole环境，任务是让杆子在小车上保持平衡。

import numpy as np
import tensorflow as tf
import gym

# 定义policy network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, n_actions):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        return self.dense2(x)

# 定义REINFORCE算法的更新参数步骤
def train_step(states, actions, rewards, model, optimizer):
    with tf.GradientTape() as tape:
        # 计算动作概率
        action_probs = model(states, training=True)
        chosen_probs = tf.reduce_sum(tf.one_hot(actions, depth=2) * action_probs, axis=1)
        # 计算损失函数
        loss = -tf.reduce_sum(tf.math.log(chosen_probs) * rewards)

    # 计算梯度并更新模型参数
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 定义主训练循环
def train_reinforce(env, model, optimizer, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        states, actions, rewards = [], [], []

        with tf.GradientTape(persistent=True) as tape:
            for t in range(1, 1000):  # 设置最大步数
                # 获取动作概率
                action_probs = model(tf.convert_to_tensor(np.reshape(state, [1,-1]), dtype=tf.float32), training=True)
                action = np.random.choice(2, p=np.squeeze(action_probs))
                # 执行动作并观察奖励
                next_state, reward, done, _ = env.step(action)

                # save state, action, and reward
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                if done:
                    break

                state = next_state

            # 计算回报
            discounted_rewards = np.cumsum(rewards[::-1])[::-1]
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            # 执行训练步骤
            train_step(np.vstack(states), np.array(actions), discounted_rewards, model, optimizer)

        if episode % 10 ==0:
            print("Episode {}: Total Reward: {}".format(episode, sum(rewards)))

# 创建CartPole环境
env = gym.make('CartPole-v1')

# create policy network and optimizer
policy_model = PolicyNetwork(n_actions=env.action_space.n)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# training model
train_reinforce(env, policy_model, optimizer)