from gym.envs.registration import register

register(
        id='simenv-v1',
        entry_point='myenv.env:SimEnv'
        )
