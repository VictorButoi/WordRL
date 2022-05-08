import itertools
from re import L
import numpy as np
from scipy.stats import entropy
import os
import gym

from wordrl.envs.wordle_env_v2_visualized import WordleEnv_v2_visualized
from wordrl.envs.wordle_env_v2_visualized import WORDS
#from wordrl.envs.wordle_env_v2_visualized import ANSWERS
from wordrl.filepaths import FILE_PATHS

MATCH_MATRIX = os.path.join(FILE_PATHS["ROOT_PATH"], "data/match_matrix.npy")


def correctness(guess, answer):
    colors = [0 for x in range(5)]
    for i in range(len(guess)):
        if guess[i] == answer[i]:
            colors[i] = 2
        elif guess[i] in answer:
            colors[i] = 1


def generate_matchings(words, answers):
    matchings = np.zeros((len(words), len(answers)))
    for i, word in enumerate(words):
        for j, answer in enumerate(answers):
            matchings[i][j] = correctness(word, answer)

    return matchings


def generate_matchings_matrix():
    matchings_matrix = generate_matchings(WORDS, WORDS)
    np.save(MATCH_MATRIX, matchings_matrix)
    return matchings_matrix


def get_distribution(words, matchings):
    distributions = np.zeros((len(words), 243))
    for word in words:
        unique, counts = np.unique(matchings, return_counts=True)

    matching_inds = np.argsort(unique)
    distributions = counts[matching_inds[:]]

    return distributions


def get_matchings(words, answers):
    matchings = MATCH_MATRIX[][]


def get_entropies(words, answers):
    distributions = get_distribution(words, get_matchings(words, answers))
    return entropy(distributions)


def get_best_guess(guess):
    best_guess =


if __name__ == "__main__":
    first_guess = "salet"
    generate_matchings_matrix()
