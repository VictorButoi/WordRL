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

from . import training
from . import losses
from . import agent
from . import make_config
from . import envs
