import itertools
from re import L
import numpy as np
from scipy.stats import entropy
import os
import gym

from wordrl.envs.wordle_env_v2_visualized import WordleEnv_v2_visualized
from wordrl.envs.wordle_env_v2_visualized import WORDS
from wordrl.envs.wordle_env_v2_visualized import ANSWERS
from wordrl.filepaths import FILE_PATHS


def correctness(guess, answer):
    colors = [0 for x in range(5)]
    for i in guess:
        if guess[i] == answer[i]:
            colors[i] = 2
        elif guess[i] in answer:
            colors[i] = 1


def get_matchings(words, answers):
    matchings = np.zeroes(len(words), len(answers))
    for i, word in enumerate(words):
        for j, answer in enumerate(answers):
            matchings[i][j] = correctness(word, answer)

    return matchings


def get_distributions(words, matchings):
    unique, counts = np.unique(matchings, return_counts=True)

    matching_inds = np.argsort(unique)
    distributions = counts[matching_inds[:]]

    return distributions


class Maximum_Entropy_Agent():
    def __init__(self):
        pass

    def get_action(self):
        action = optimal_guess
        return action
