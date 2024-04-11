# -*- coding: utf-8 -*-
# @Time : 2024/4/7 22:19
# @Author : yysgz
# @File : CartPole_REINFORCE.py
# @Project : RL_models
# @Description :

import pathlib  # 用于处理文件路径的模块
from dataclasses import asdict  # 用于将数据类dataclass的实例转换为字典dict

import gymnasium as gym
import numpy as np
import pandas as pd

import sys
sys.path.append(r'C:\Users\yysgz\OneDrive - Macquarie University\Desktop\RL_models\CartPole_Q_Learning\cartpole')

from cartpole.agents import Agent, QLearningAgent
from cartpole.entities import Action, EpisodeHistory, EpisodeHistoryRecord, Observation, Reward
from cartpole.plotting import EpisodeHistoryMatplotlibPlotter

# 用于记录当前时间步的信息。
def log_timestep(index: int, action: Action, reward: Reward, observation: Observation) -> None:
    'Log the information about the current timestep results.'
    format_string = "  ".join(
        ['Timestep: {0:3d}',
         'Action: {1:2d}',
         'Reward: {2:5.1f}',
         'Cart Position: {3:6.3f}',
         'Cart Velocity: {4:6.3f}',
         'Angle: {5:6.3f}',
         'Tip Velocity: {6:6.3f}',]
    )
    print(format_string.format(index, action, reward, *observation))

def run_agent(agent: Agent, env: gym.Env, verbose: bool=False) -> EpisodeHistory:
    'Run an intelligent cartpole agent in a cartpole environment, capturing the episode history.'
    '它接受agent对象和env作为输入，并返回一个EpisodeHistory对象，记录了每个episode历史。'
    max_episodes_to_run = 5000
    max_timesteps_per_episode = 200

    # The environment is solved if we can survive for avg. 195 timesteps across 100 episodes.
    goal_avg_episode_length = 195
    goal_consecutive_episodes = 100

    episode_history = EpisodeHistory(max_timesteps_per_episode=200, goal_avg_episode_length=goal_avg_episode_length,
                                     goal_consecutive_episodes=goal_consecutive_episodes)
    episode_history_plotter = None

    if verbose:
        episode_history_plotter = EpisodeHistoryMatplotlibPlotter(history=episode_history, visible_episode_count=200)  # how many most recent episodes to fit on a single plot.
        episode_history_plotter.create_plot()

    # Main simulation/learning loop.
    print('Running the environment. To stop, press Ctrl+C.')
    try:
        for episode_index in range(max_episodes_to_run):
            state, _ = env.reset()  # [ 0.02532753 -0.24024986 -0.00714913  0.2850354 ]
            action = agent.begin_episode(state)

            for timestep_index in range(max_timesteps_per_episode):
                # perform the action and observe the new state.
                next_state, reward, terminated, _, _ = env.step(action)  # reward: 1.0; terminated: False/False

                # Log the current state.
                if verbose:
                    log_timestep(timestep_index, action, reward, next_state)

                # If the episode has ended prematurely, penalized the agent.
                is_successful = timestep_index >= max_timesteps_per_episode - 1
                if terminated and not is_successful:
                    reward = -max_episodes_to_run
                # Get the next action from the learner, given our new state. 更新Q table
                action = agent.act(next_state, reward)

                # Record this episode to the history and check if the goal has been reached.
                if terminated or is_successful:  # 注意这里的or 逻辑判断！
                    print(f"Episode: {episode_index}, "
                          f"finished after {timestep_index + 1} timesteps.")

                    episode_history.record_episode(
                        EpisodeHistoryRecord(
                            episode_index=episode_index,
                            episode_length=timestep_index + 1,
                            is_successful=is_successful,
                        )
                    )
                    if verbose and episode_history_plotter:  # 始终为ture！！
                        episode_history_plotter.update_plot()
                    if episode_history.is_goal_reached():
                        print(f"SUCCESS: Goal reached after {episode_index + 1} episodes!")
                        return episode_history
                    break
        print(f"FAILURE: Goal not reached after {max_episodes_to_run} episodes.")
    except KeyboardInterrupt:
        print('WARNING: Terminated by user request.')

    return episode_history

def save_history(history: EpisodeHistory, experiment_dir: str) -> pathlib.Path:
    '''
    Save the episode history to a CSV file.
    :param history: history to save.
    :param experiment_dir: Name of directory to save the history to. Will be created if nonexistent.
    :return:
    '''

    experiment_dir_path = pathlib.Path(experiment_dir)
    experiment_dir_path.mkdir(parents=True, exist_ok=True)

    file_path = experiment_dir_path / "episode_history.csv"
    record_dicts = (asdict(record) for record in history.all_records())
    dataframe = pd.DataFrame.from_records(record_dicts, index='episode_index')
    dataframe.to_csv(file_path, header=True)
    print(f"Episode history saved to {file_path}")
    return file_path

if __name__ == "__main__":
    # sys.argv 将会是 ['myscript.py', 'arg1', 'arg2']。第一个元素是脚本的名称，其余的元素是传递给脚本的参数。
    # verbose = len(sys.argv) > 1 and sys.argv[1] == "--verbose"  # 设置了是否输出详细信息的表示verbose
    verbose = True
    random_state = np.random.RandomState(seed=0)


    env = gym.make("CartPole-v1", render_mode='human' if verbose else None)
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=0.95,
        exploration_rate=0.5,
        exploration_decay_rate=0.99,
        random_state=random_state,
    )

    episode_history = run_agent(agent=agent, env=env, verbose=verbose)
    save_history(episode_history, experiment_dir='experiment-results')