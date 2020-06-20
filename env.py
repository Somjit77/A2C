import gym
import gym_minigrid
import numpy as np


class Environment:
    def __init__(self, game_name, seed):
        """Initialize Environment"""
        self.game_name = game_name
        self.env = gym.make(self.game_name)
        self.env.seed(seed)
        self.number_of_actions = self.env.action_space.n
        if 'MiniGrid' in self.game_name:
            self.state_space = self.env.observation_space['image']
        else:
            self.state_space = self.env.observation_space

    def process_state(self, observation):
        """Pre-process state if required"""
        if 'MiniGrid' in self.game_name:
            return np.array(observation['image'], dtype='float32')  # Using only image as state (7x7x3)
        else:
            return observation

    def reset(self):
        if 'MiniGrid' in self.game_name:
            return self.process_state(self.env.reset())
        else:
            return self.env.reset()

    def step(self, action):
        if 'MiniGrid' in self.game_name:
            next_state, reward, done, info = self.env.step(action)
            return self.process_state(next_state), reward, done, info
        else:
            return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
