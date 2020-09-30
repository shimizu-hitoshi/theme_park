import torch

DEBUG = False # True # False

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, device):
        self.observations = torch.zeros(num_steps + 1, num_processes, obs_shape).to(device)
        self.masks        = torch.ones(num_steps + 1,  num_processes, 1).to(device)
        self.rewards      = torch.zeros(num_steps,     num_processes, 1).to(device)
        self.actions      = torch.zeros(num_steps,     num_processes, 1).long().to(device)
        self.returns      = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.index = 0

    def insert(self, current_obs, action, reward, mask, NUM_ADVANCED_STEP):
        if DEBUG: print(current_obs.shape)
        if DEBUG: print(action.shape)
        if DEBUG: print(reward.shape)
        if DEBUG: print(mask.shape)

        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        # print(self.actions[self.index,:,0].shape, action.shape)
        # self.actions[self.index,:,0].copy_(action)
        # print(self.actions[self.index].shape, action.shape)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, GAMMA):
        # print(self.returns[-1].shape, next_value.shape)
        # self.returns[-1,:,0] = next_value
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]

