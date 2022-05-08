import wordrl as wdl
import pkg_resources
import gym
import numpy as np
import random
import os
from typing import Optional
import pygame

WORDLE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
WORD_LENGTH = 5
GAME_LENGTH = 6

white = (255, 255, 255)
black = (0, 0, 0)
grey = (69, 69, 69)

# FROM OFFICIAL WORDLE GAME
green = (120, 177, 90)
yellow = (253, 203, 88)


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


class WordleEnv_v2_visualized(gym.Env):
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

    def __init__(self, word_file):
        super(WordleEnv_v2_visualized, self).__init__()
        WORDS_PATH = os.path.join(
            wdl.filepaths.FILE_PATHS["ROOT_PATH"], f"data/{word_file}")
        self.words = get_words(WORDS_PATH)
        self.max_turns = GAME_LENGTH
        self.action_space = gym.spaces.Discrete(len(self.words))
        self.observation_space = gym.spaces.MultiDiscrete(
            [GAME_LENGTH] + [2] * len(WORDLE_CHARS) + [2] * 3 * WORD_LENGTH * len(WORDLE_CHARS))

        # for the visualizer
        self.guesses = []
        self.colors = []

        # pygame stuff
        self.screen = None
        self.clock = None
        self.isopen = True

        self.done = True
        self.state: np.ndarray = None

    def step(self, action):
        assert self.action_space.contains(action)
        self.guesses.append(self.words[action])

        action = self.words[action]
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
                self.colors.append(green)
                # c is at position i, all other chars are not at position i
                state[offset + 3 * i:offset + 3 * i + 3] = [0, 0, 1]
                for ocint in range(len(WORDLE_CHARS)):
                    if ocint != cint:
                        oc_offset = 1 + len(WORDLE_CHARS) + \
                            ocint * WORD_LENGTH * 3
                        state[oc_offset + 3 * i: oc_offset +
                              3 * i + 3] = [1, 0, 0]
            elif c in self.goal_word:
                self.colors.append(yellow)
                # c is not at position i, but could be at other positions
                state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
            else:
                self.colors.append(grey)
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
        self.guesses = []
        self.colors = []

        #print("Goal word is : " + self.goal_word)
        return self.state.copy()

    def render(self, mode="human"):
        # print(self.guesses)

        # GAME SETTINGS
        NUM_ROWS = 6
        NUM_COLUMNS = 5

        # GRAPHICS CONSTANTS, add 100 for spacing
        BOX_SIZE = 100
        BOX_SPACING = 10
        SCREEN_SPACING = 200
        SCREEN_HEIGHT = NUM_ROWS*BOX_SIZE + \
            (NUM_ROWS - 1)*BOX_SPACING + SCREEN_SPACING
        SCREEN_WIDTH = NUM_COLUMNS*BOX_SIZE + \
            (NUM_COLUMNS - 1)*BOX_SPACING + SCREEN_SPACING

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # PYGAME OVERHEAD
        #screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Wordle Knockoff")
        turn = 0
        fps = 5
        timer = pygame.time.Clock()
        huge_font = pygame.font.Font("freesansbold.ttf", 56)
        font_x = 40
        font_y = 48
        game_over = False
        text_surface = huge_font.render(
            "GOAL WORD: "+self.goal_word, False, white)

        # GRAPHICS SETUP
        board = [[" " for _ in range(NUM_COLUMNS)] for _ in range(NUM_ROWS)]

        def place_letter(x, y, x_idx, box_size, word):
            piece_text = huge_font.render(word[x_idx], True, white)
            x_offset = (box_size - font_x)/2
            y_offset = (box_size - font_y)/2
            self.screen.blit(piece_text, (x + x_offset, y + y_offset))

        def draw_board(board,
                       sox,
                       soy,
                       box_size,
                       x_space,
                       y_space,
                       do_fill=0):
            height = len(board)
            width = len(board[0])

            for row in range(height):
                for col in range(width):
                    # convention for drawing is [x, y, width, height], boarder, rounding
                    x = col*box_size + sox + col*x_space
                    y = row*box_size + soy + row*y_space
                    if row < len(self.guesses):
                        pygame.draw.rect(self.screen, self.colors[row*5+col], [
                            x, y, box_size, box_size], do_fill)
                        place_letter(x, y, col, box_size, self.guesses[row])
                    else:
                        pygame.draw.rect(self.screen, grey, [
                            x, y, box_size, box_size], do_fill)

        x_offset = (SCREEN_WIDTH - (NUM_COLUMNS*BOX_SIZE +
                    (NUM_COLUMNS - 1)*BOX_SPACING))/2
        y_offset = (SCREEN_HEIGHT - (NUM_ROWS*BOX_SIZE +
                    (NUM_ROWS - 1)*BOX_SPACING))/2
        if self.state[0] == 5:
            self.screen.fill(black)
            self.screen.blit(text_surface, (0, 0))

        timer.tick(fps)
        draw_board(board, sox=x_offset, soy=y_offset, box_size=BOX_SIZE,
                   x_space=BOX_SPACING, y_space=BOX_SPACING)
        # updates the screen
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
