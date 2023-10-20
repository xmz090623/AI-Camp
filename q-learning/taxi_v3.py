# -*- coding: utf-8 -*-
import logging
import gym
import numpy as np
import sys
import tqdm

LEARNING_RATE = 0.9
REWARD_DISCOUNT_RATE = 0.8
DISCOVERY_RATE = 1.0
DISCOVERY_RATE_MINIMAL = 0.1
DISCOVERY_RATE_DAMPING = 0.005

EPISODES = 10_000
MAX_STEPS = 50
ROUNDS = 3

LOG_LEVEL = "DEBUG"

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
        new_discovery_rate = np.exp(-self._discovery_rate_damping * self._discovery_rate)
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
            for s in range(max_step):
                action = self.get_action(random=random, training=False, state=state)
                new_state, reward, done, _, _ = self._env.step(action)
                records += reward
                self._env.render()
                if done:
                    break
                state = new_state
            self.logger.info(f"Round {r} record: {records}")

        self.logger.info(f"Total records: {records}")

    def training_q_learning(self, episodes: int = 10_000, max_steps: int = 99):
        self.logger.info(f"Training Q-Learning, with episodes {episodes} and max_step {max_steps} per episode")
        ui_env = gym.wrappers.RecordEpisodeStatistics(self._env, deque_size=episodes)
        for _ in tqdm.tqdm(range(episodes)):
            state = ui_env.reset()[0]

            for s in range(max_steps):
                action = self.get_action(training=True, state=state)
                new_state, reward, done, _, _ = ui_env.step(action)

                self._q_table[state, action] = self._q_table[state, action] + self._learning_rate * (
                        reward + self._reward_discount_rate * np.max(self._q_table[new_state, :]) - self._q_table[state, action])
                if done:
                    break

                state = new_state

            self.refresh_discovery_rate()


if __name__ == "__main__":
    env_play = gym.make('Taxi-v3', render_mode="human")
    taxi_v3_game = TaxiV3(env=env_play,
                          learning_rate=LEARNING_RATE,
                          reward_discount_rate=REWARD_DISCOUNT_RATE,
                          discovery_rate=DISCOVERY_RATE,
                          discovery_rate_minimal=DISCOVERY_RATE_MINIMAL,
                          discovery_rate_damping=DISCOVERY_RATE_DAMPING,
                          logger=LOG)

    LOG.info(f"Drive taxi in random environment, with {ROUNDS} rounds, and each round {MAX_STEPS} steps")
    taxi_v3_game.play(random=True, max_step=15, rounds=1)

    env_training = gym.make('Taxi-v3', render_mode="rgb_array")
    taxi_v3_game.env = env_training
    taxi_v3_game.training_q_learning(episodes=EPISODES)

    LOG.info(f"Drive taxi in Trained Q-Learning environment, with {ROUNDS} rounds, and each round {MAX_STEPS} steps")
    taxi_v3_game.env = env_play
    taxi_v3_game.play(random=False, max_step=MAX_STEPS, rounds=20)

    taxi_v3_game.env.close()
