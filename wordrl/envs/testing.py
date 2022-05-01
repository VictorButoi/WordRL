import gym
from gym import spaces
import numpy as np
import pkg_resources
import random
from typing import Optional

filename = pkg_resources.resource_filename(
    'WordRL',
    'data/random_words.txt'
)


def encodeToStr(encoding):
    string = ""
    for enc in encoding:
        string += chr(ord('a') + enc)
    return string


def strToEncode(lines):
    encoding = []
    for line in lines:
        assert len(line.strip()) == 5  # Must contain 5-letter words for now
        encoding.append(tuple(ord(char) - 97 for char in line.strip()))
    return encoding


with open(filename, "r") as f:
    WORDS = strToEncode(f.readlines())

print(WORDS)
