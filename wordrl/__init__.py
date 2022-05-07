from . import wordle
from . import a2c
from . import dqn
from . import filepaths
from . import envs
from . import make_config

from gym.envs.registration import register

register(
    id='Wordle-v2-10',
    entry_point='wordrl.envs:WordleEnv_v2',
    max_episode_steps=200,
)

register(
    id='Wordle-v2-10-visualized',
    entry_point='wordrl.envs:WordleEnv_v2_visualized',
    max_episode_steps=200,
)
