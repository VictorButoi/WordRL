"""
Usage: 
  main_kelly_v2.py random [options]
  main_kelly_v2.py dqn [options]

options:
  -r --render                 render visualizer 

"""

import gym
import gym_wordle
from gym_wordle.exceptions import InvalidWordException
from gym_wordle.agents.agent_random_v2 import Agent
from docopt import docopt


def command_line_parser():
    args = docopt(__doc__)

    if args['random']:
        #run random 

    elif args['dqn']:
        #run dqn 
    else: 
        raise RuntimeError("Invalid CLI Argument")



#Add in training for other agents 

# RL parameters
resume_from_checkpoint = None
initialize_winning_replays = None
#env = "WordleEnv100-v0"
deep_q_network = 'SumChars'

# Training parameters
max_epochs = 5000
checkpoint_every_n_epochs = 1000
replay_size = 1000
hidden_size = 256
sync_rate = 100
lr = 1.e-3
weight_decay = 1.e-5
last_frame_cutoff = 0.8
max_eps = 1
min_eps = 0.01
episode_length = 512
batch_size = 512

env = gym.make('Wordle-v2-10-visualized')

random_agent = Agent()

obs = env.reset()
done = False
while not done:
    while True:
        try:
            # make a random guess
            act = random_agent.action(env.action_space, obs)

            # take a step
            obs, reward, done, info = env.step(act)
            break
        except InvalidWordException:
            pass

    env.render()
