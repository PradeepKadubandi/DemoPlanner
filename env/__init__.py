from gym.envs.registration import register

register(
    id='simple-reacher-v0',
    entry_point='env.reacher:SimpleReacherEnv',
    kwargs={},
)
