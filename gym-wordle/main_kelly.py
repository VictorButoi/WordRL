import gym
import gym_wordle
from gym_wordle.exceptions import InvalidWordException
from gym_wordle.agents.agent_random import Agent

env = gym.make('Wordle-v0')

random_agent = Agent()

obs = env.reset()
done = False
while not done:
    while True:
        try:
            # make a random guess
            act = random_agent.action(env.action_space, obs)

            # take a step
            obs, reward, done, _ = env.step(act)
            break
        except InvalidWordException:
            pass

    env.render()
