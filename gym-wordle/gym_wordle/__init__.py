from gym.envs.registration import register

register(
    id='Wordle-v0',
    entry_point='gym_wordle.envs:WordleEnv',
    reward_threshold=1.0
)

register(
    id='Wordle-v2-10',
    entry_point='gym_wordle.envs:WordleEnv_v2',
    max_episode_steps=200,
)
