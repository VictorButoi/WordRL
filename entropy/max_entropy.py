import itertools
import numpy as np
import os
import gym

from typing import Optional
from re import L
from scipy.stats import entropy

import wordrl as wdl


def get_words(filename, limit: Optional[int] = None):
    """
    Takes a .txt file of words spaced by newlines and loads the words into a list of strings
    Optional int argument limit: specify to take only the first words in the list up to limit
    """

    with open(filename, "r") as f:
        words = [line.strip().upper() for line in f.readlines()]
        if not limit:
            return words
        else:
            return words[:limit]


MATCH_MATRIX = os.path.join(
    wdl.filepaths.FILE_PATHS["ROOT_PATH"], "data/match_matrix.npy")
WORDS_PATH = os.path.join(
    wdl.filepaths.FILE_PATHS["ROOT_PATH"], f"data/big_wordle_words.txt")
WORDS = get_words(WORDS_PATH)


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


def get_matchings(words, answers):

    matchings = MATCH_MATRIX[][]


def get_distributions(words, answers):
    matchings = get_matchings(words, answers)
    distributions = np.zeros((len(words), 243))
    for i in range(len(words)):
        unique, counts = np.unique(matchings[i], return_counts=True)
        distributions[i][ternary_to_decimal(unique)] = counts

    return distributions


def ternary_to_decimal(ternary):
    decimal = 0
    for i in range(len(ternary)):
        decimal += (3**i)*ternary[i]
    return decimal


def get_entropies(words, answers):
    distributions = get_distributions(words, answers)
    return entropy(distributions, axis=0)


def get_best_guess(words, answers):
    entropy = get_entropies(words, answers)
    best_guess_index = np.argmax(entropy, axis=0)

    return words[best_guess_index]


def play_game(words_list, answers_list, goal_word):
    pass


if __name__ == "__main__":
    generate_matchings_matrix()
