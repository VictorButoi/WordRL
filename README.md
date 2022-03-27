# gym-wordle

setup and data files copied from  [zach-lawless/gym-wordle](https://github.com/zach-lawless/gym-wordle)

## About
To play one game of Wordle using a random agent run `main_kelly_v2.py`

## Methodology

### Words
The list of all possible guesses is stored in `gym_wordle/data/5_words.txt`

### Environment

#### Observation Space
A length 417 MultiDiscrete space representing a 1D int array where:
* index[0] = number of guesses remaining
* [1..27] = binary, whether each of the 26 letters has been guessed
* [[status, status, status, status, status] for _ in WORDLE_CHARS] = binary, where status has codes
    * [1, 0, 0] -- char is definitely not in this spot
    * [0, 1, 0] -- char may be in this spot
    * [0, 0, 1] -- char is in this spot

#### Action Space
In order to step through the environment, supply `env.step()` with an integer corresponding to the index of the desired guess in the word list

#### Rewards and Exit Criteria

The reward structure for the game is below:
* If the player guesses the hidden word in six guesses or fewer, the environment returns `reward=10` and `done=True`.
* If the player hasn't guessed the hidden word but still has guesses remaining, the environment returns `reward=0` and `done=False`.
* If the player is out of guesses and hasn't guessed the hidden word, the environment returns `reward=-10` and `done=True`.

#### Render
Right now just prints the state space
