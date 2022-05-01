import wordrl as wdl
import pkg_resources
import gym
import numpy as np
import random
import os
from typing import Optional

WORDLE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
WORD_LENGTH = 5
GAME_LENGTH = 6


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


WORDS_PATH = os.path.join(wdl.paths.DATA_PATH, "5_words.txt")

WORDS = get_words(WORDS_PATH, 100)


class WordleEnv_v2(gym.Env):
    """
    Bare Bones Wordle Environment

    Action Space: A Discrete space with length corresponding to all the possible 5 letter word guesses in Wordle (number of words in file located at WORDS_PATH)

    Observation Space (i.e. State Space): A length 417 MultiDiscrete space representing a 1D int array where:
        index[0] = number of guesses remaining
        [1..27] = binary, whether each of the 26 letters has been guessed
        [[status, status, status, status, status] for _ in WORDLE_CHARS] = binary
            where status has codes
            [1, 0, 0] -- char is definitely not in this spot
            [0, 1, 0] -- char may be in this spot
            [0, 0, 1] -- char is in this spot

    Reward: For now the reward will be a placeholder with simply a 10 for if the word was guessed correctly, -10 if the agent ran out of guesses, and 0 otherwise

    """

    def __init__(self):
        super(WordleEnv_v2, self).__init__()
        self.words = WORDS
        self.action_space = gym.spaces.Discrete(len(self.words))
        self.observation_space = gym.spaces.MultiDiscrete(
            [GAME_LENGTH] + [2] * len(WORDLE_CHARS) + [2] * 3 * WORD_LENGTH * len(WORDLE_CHARS))
        self.max_turns = GAME_LENGTH

        self.done = True
        self.state: np.ndarray = None

    def step(self, action):
        assert self.action_space.contains(action)

        action = WORDS[action]
        if self.done:
            raise ValueError(
                "The game is already done (the environment returned done = True). Call reset() before attempting to call step() again."

            )
        state = self.state.copy()

        # update the state
        state[0] -= 1
        for i, c in enumerate(action):  # iterate over letters in the word
            cint = ord(c) - ord(WORDLE_CHARS[0])  # get the index of the char
            offset = 1 + len(WORDLE_CHARS) + cint * WORD_LENGTH * 3
            state[1 + cint] = 1  # letter has now been guessed
            if self.goal_word[i] == c:
                # c is at position i, all other chars are not at position i
                state[offset + 3 * i:offset + 3 * i + 3] = [0, 0, 1]
                for ocint in range(len(WORDLE_CHARS)):
                    if ocint != cint:
                        oc_offset = 1 + len(WORDLE_CHARS) + \
                            ocint * WORD_LENGTH * 3
                        state[oc_offset + 3 * i: oc_offset +
                              3 * i + 3] = [1, 0, 0]
            elif c in self.goal_word:
                # c is not at position i, but could be at other positions
                state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
            else:
                # c is not in the goal word
                state[offset:offset+3*WORD_LENGTH] = [1, 0, 0] * WORD_LENGTH

        self.state = state

        reward = 0
        if action == self.goal_word:
            self.done = True
            reward = 10
        elif self.state[0] == 0:
            self.done = True
            reward = -10

        return self.state.copy(), reward, self.done, {'goal_id': self.goal_word}

    def reset(self):
        self.done = False
        self.goal_word = random.choice(self.words)
        self.state = np.array([GAME_LENGTH] + [0] * len(WORDLE_CHARS) +
                              [0, 1, 0] * WORD_LENGTH * len(WORDLE_CHARS), dtype=np.int32)

        return self.state.copy()

    def render(self, mode="human"):
        print(self.state)
