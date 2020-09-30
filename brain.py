import torch
from torch import optim

class Brain(object):
    def __init__(self, actor_critic,config):
        self.actor_critic = actor_critic
        self.optimizer    = optim.Adam(self.actor_critic.parameters(), lr=0.001)
        self.value_loss_coef   = float( config['TRAINING']['loss_coef'] )
        self.entropy_coef      = float( config['TRAINING']['entropy_coef'] )
        self.max_grad_norm     = float( config['TRAINING']['max_grad_norm'] )
        self.num_advanced_step = config.getint('TRAINING','num_advanced_step')
        self.num_parallel     = config.getint( 'TRAINING','num_parallel')

    def update(self, rollouts):
        obs_shape         = rollouts.observations.size()[2:]
        value_loss_coef   = self.value_loss_coef
        entropy_coef      = self.entropy_coef
        max_grad_norm     = self.max_grad_norm
        num_advanced_step = self.num_advanced_step
        num_parallel = self.num_parallel

        # print(rollouts.observations[:-1].view(-1, 2414))
        # print(rollouts.observations[:-1].shape)
        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
                rollouts.observations[:-1].view(-1, *obs_shape),
                rollouts.actions.view(-1, 1))
        # print(values.shape)
        values           = values.view(num_advanced_step, num_parallel, 1)
        action_log_probs = action_log_probs.view(num_advanced_step, num_parallel, 1)
        advantages       = rollouts.returns[:-1] - values
        value_loss       = advantages.pow(2).mean()
        action_loss      = -(action_log_probs * advantages.detach()).mean()
        # total_loss       = (value_loss * value_loss_coef + action_gain - entropy * entropy_coef)
        total_loss       = (value_loss * value_loss_coef + action_loss - entropy * entropy_coef)

        # total_loss.requires_grad = True

        self.actor_critic.train() # train mode
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)

        self.optimizer.step()
        return value_loss, action_loss, total_loss, entropy 
