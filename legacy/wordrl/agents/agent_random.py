from gym_wordle.envs.wordle_env_v2 import WORDS
import numpy as np
import gym
import random


class Agent():
    """
    Agent for our WordRL Gym environment

    """

    def __init__(self, env='env', name='Random'):
        """ Initialization of agent """
        self.Action = WORDS

    def action(self, action_space, observation):
        """ Randomly chooses an action (a word) from the list of possible word guesses """

        _ = observation  # random agent, independent of observation

        action = np.asarray(random.choice(self.Action))
        action.flatten()

        return action
