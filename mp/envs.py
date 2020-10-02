import gym
import torch
import copy
from model import ActorCritic
from brain import Brain
from storage import RolloutStorage
import numpy as np
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

# from baselines.common.vec_env import VecEnvWrapper
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
# from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

# See https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py
def make_env(env_name, seed, env_id, datadir, config, R_base):
    # print("R_base @ make_env",R_base)
    def _thunk():
        # print("seed and rank: ",seed, rank)
        env = gym.make(env_name)
        env.seed(seed, env_id, datadir, config, R_base)
        # obs_shape = env.observation_space.shape
        return env
    return _thunk

# def make_vec_envs(env_name, seed, num_process, device, datadirs, training_targets, config):
# def make_vec_envs(env_name, seed, num_parallel, device, datadirs, config, R_base=(None,None)):
def make_vec_envs(env_name, seed, num_parallel, device, datadirs, config, R_base=None):
    # print(len(datadirs), len(training_targets), len(fixed_agents))
    print(len(datadirs), datadirs)
    # print(dict_target)
    # print("R_base @ make_vec_envs",R_base)
    envs = [
            # make_env(env_name, seed, i, datadirs[i], training_targets[i], config)
            make_env(env_name, seed, i, datadirs[i], config, R_base)
            for i in range(num_parallel) # i: env_id ということにする
            # for i in range(num_parallel * len(training_targets))
            # for i in range(num_process)
            ]
    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    return envs

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs    = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def set_t_open(self, T_open):
        self.venv.set_t_open(T_open)
