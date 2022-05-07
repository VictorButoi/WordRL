from typing import Tuple, List

# wordrl imports
import wordrl as wdl

from . import module
from . import agent


def load_from_checkpoint(
        checkpoint: str
) -> Tuple[module.AdvantageActorCritic, agent.GreedyActorCriticAgent, wdl.wordle.wordle.WordleEnvBase]:
    """
    :param checkpoint:
    :return:
    """
    model = module.AdvantageActorCritic.load_from_checkpoint(checkpoint)
    agent = agent.GreedyActorCriticAgent(model.net)
    env = model.env

    return model, agent, env


def suggest(
        agent: agent.GreedyActorCriticAgent,
        env: wdl.wordle.wordle.WordleEnvBase,
        sequence: List[Tuple[str, List[int]]],
) -> str:
    """
    Given a list of words and masks, return the next suggested word

    :param agent:
    :param env:
    :param sequence: History of moves and outcomes until now
    :return:
    """
    state = env.reset()
    for word, mask in sequence:
        word = word.upper()
        assert word in env.words, f'{word} not in allowed words!'
        assert all(i in (0, 1, 2) for i in mask)
        assert len(mask) == 5

        state = wdl.wordle.state.update_from_mask(state, word, mask)

    return env.words[agent(state, "cpu")[0]]


def goal(
        agent: agent.GreedyActorCriticAgent,
        env: wdl.wordle.wordle.WordleEnvBase,
        goal_word: str,
) -> Tuple[bool, List[Tuple[str, int]]]:
    state = env.reset()
    try:
        env.set_goal_word(goal_word.upper())
    except:
        raise ValueError("Goal word", goal_word, "not found in env words!")

    outcomes = []
    win = False
    for i in range(env.max_turns):
        action = agent(state, "cpu")[0]
        state, reward, done, _ = env.step(action)
        outcomes.append((env.words[action], reward))
        if done:
            if reward >= 0:
                win = True
            break

    return win, outcomes
