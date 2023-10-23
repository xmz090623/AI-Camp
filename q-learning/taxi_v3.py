# -*- coding: utf-8 -*-
import gym
import numpy as np
import sys
import time
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

LEARNING_RATE = 0.9
REWARD_DISCOUNT_RATE = 0.8
DISCOVERY_RATE = 1.0
DISCOVERY_RATE_MINIMAL = 0.05
DISCOVERY_RATE_DAMPING = 0.0001

EPISODES = 15_000
PLAY_STEPS = 50
ROUNDS = 3

LOG_LEVEL = "INFO"

logging.basicConfig()
LOG = logging.getLogger()
LOG.setLevel(LOG_LEVEL)


class TaxiV3(object):
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float,
                 reward_discount_rate: float,
                 discovery_rate: float,
                 discovery_rate_minimal: float,
                 discovery_rate_damping: float,
                 logger: logging.Logger = logging.getLogger()
                 ):
        self._env = env
        self._learning_rate = learning_rate
        self._reward_discount_rate = reward_discount_rate
        self._discovery_rate = discovery_rate
        self._discovery_rate_minimal = discovery_rate_minimal
        self._discovery_rate_damping = discovery_rate_damping
        self._q_table = np.zeros((self._env.observation_space.n, self._env.action_space.n))
        self.logger = logger

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

    def refresh_discovery_rate(self):
        new_discovery_rate = self._discovery_rate - self._discovery_rate_damping
        self._discovery_rate = max(self._discovery_rate_minimal, new_discovery_rate)

    def get_action(self, random=True, training=False, state=None):
        if training:
            if np.random.uniform(0, 1) < self._discovery_rate:
                return self._env.action_space.sample()
            else:
                return np.argmax(self._q_table[state, :])

        if random:
            return self._env.action_space.sample()

        return np.argmax(self._q_table[state, :])

    # random True always return action_space.sample(), which means play without q-learning algorithm
    def play(self, random: bool = True, max_step: int = 30, rounds: int = 3):
        if max_step < 10:
            self.logger.error(f"Setting max_step less than 10 doesn't make sense. Current value: {max_step}")
            sys.exit(1)

        records = 0
        # State after reset: (304, {'prob': 1.0, 'action_mask': array([1, 1, 0, 0, 0, 0], dtype=int8)})
        for r in range(rounds):
            state, action = self._env.reset()
            step = 0
            for s in range(max_step):
                action = self.get_action(random=random, training=False, state=state)
                new_state, reward, done, _, _ = self._env.step(action)
                records += reward
                self._env.render()
                step = s
                if done:
                    break
                state = new_state
            self.logger.info(f"Round {r+1} record: {records}; Steps: {step+1}")

    def training_q_learning(self, episodes: int = 10_000, allow_steps: int = 99):
        bar_number = 10
        sample_rate = 50
        step_sample_list = []
        self.logger.info(f"Training Q-Learning, with episodes {episodes} and max_step {allow_steps} per episode")
        # ui_env = gym.wrappers.RecordEpisodeStatistics(self._env, deque_size=episodes)
        for b in range(bar_number):
            step_cost_list = []
            with tqdm(total=int(episodes/bar_number), desc=f"Episode {b}", unit="episode") as progress_bar:
                for e in range(int(episodes/bar_number)):
                    state = self._env.reset()[0]

                    for s in range(allow_steps):
                        action = self.get_action(training=True, state=state)
                        new_state, reward, done, _, _ = self._env.step(action)

                        self._q_table[state, action] = self._q_table[state, action] + self._learning_rate * (
                            reward + self._reward_discount_rate * np.max(self._q_table[new_state, :]) - self._q_table[state, action])
                        if done:
                            break
                        state = new_state

                    step_cost_list.append(s + 1)
                    self.refresh_discovery_rate()

                    if (e + 1) % sample_rate == 0:
                        avg_step = np.mean(step_cost_list[-sample_rate:])
                        progress_bar.set_postfix({
                            "episode": f"{int(episodes / bar_number * b + e + 1)}",
                            "average_step_cost": "{0:.1f}".format(avg_step),
                            "discovery_rate": '{0:.1f}'.format(self._discovery_rate)
                        })
                        step_sample_list.append(avg_step)
                    progress_bar.update(1)

        episodes_list = list(range(int(episodes/sample_rate)))
        plt.plot(episodes_list, step_sample_list)
        plt.xlabel(f"{sample_rate} Episodes")
        plt.ylabel(f"Average Steps in {sample_rate} episodes")
        plt.title('Q-Table on {}'.format('Taxi V3'))
        plt.show(block=False)
        plt.pause(5)
        plt.close()


if __name__ == "__main__":
    env_play = gym.make('Taxi-v3', render_mode="human")
    taxi_v3_game = TaxiV3(env=env_play,
                          learning_rate=LEARNING_RATE,
                          reward_discount_rate=REWARD_DISCOUNT_RATE,
                          discovery_rate=DISCOVERY_RATE,
                          discovery_rate_minimal=DISCOVERY_RATE_MINIMAL,
                          discovery_rate_damping=DISCOVERY_RATE_DAMPING,
                          logger=LOG)

    LOG.info(f"Drive taxi in random environment, with 1 round, and each round {PLAY_STEPS} steps")
    taxi_v3_game.play(random=True, max_step=PLAY_STEPS, rounds=1)

    env_training = gym.make('Taxi-v3', render_mode="rgb_array")
    taxi_v3_game.env = env_training
    taxi_v3_game.training_q_learning(episodes=EPISODES)

    LOG.info(f"Drive taxi in Trained Q-Learning environment, with {ROUNDS} rounds, and each round {PLAY_STEPS} steps")
    taxi_v3_game.env = env_play
    taxi_v3_game.play(random=False, max_step=PLAY_STEPS, rounds=ROUNDS)

    time.sleep(5)
    taxi_v3_game.env.close()
