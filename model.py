import torch
import torch.nn as nn
import torch.nn.functional as F
def init(module, gain):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module

class ActorCritic(nn.Module):
    def __init__(self, n_in, n_out):
        super(ActorCritic, self).__init__()

        def init_(module): return init(module, gain=nn.init.calculate_gain('relu'))

        mid_io = 128
        self.linear1 = nn.Linear(n_in, mid_io)
        self.linear2 = nn.Linear(mid_io, mid_io)
        self.linear3 = nn.Linear(mid_io, mid_io)

        self.actor   = nn.Linear(mid_io, n_out)
        # nn.init.normal_(self.actor.weight, 0.0, 1.0)
        self.critic  = nn.Linear(mid_io, 1)
        # nn.init.normal_(self.critic.weight, 0.0, 1.0)

    def forward(self, x):
        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        h3 = F.relu(self.linear3(h2))

        critic_output = self.critic(h3)
        actor_output  = self.actor(h3)
        return critic_output, actor_output

    # def set_edges(self, edges):
    #     self.num_edges = edges.num_obsv_edge
    #     self.num_goals = edges.num_obsv_goal

    # def legal_actions(self, obs):
    #     x = obs[:,self.num_edges:(self.num_edges+self.num_goals)] # 状態の冒頭に道路上人数，次に残容量がある想定
    #     ret = torch.where( x > 0 )
    #     if ret[0].shape[0] == 0:
    #         ret = torch.where( x == 0 )
    #     # print("ret",ret)
    #     return ret

    def act(self, x, flg_greedy=False):
        value, actor_output = self(x)
        # if flg_legal: # 空いている避難所のみを誘導先候補にする
        #     ret = torch.zeros(x.shape[0],1)
        #     legal_actions = self.legal_actions(x)
        #     for i in range(x.shape[0]):
        #         idxs = legal_actions[1][legal_actions[0]==i]
        #         if flg_greedy:
        #             action_probs = F.softmax(actor_output[i,idxs], dim=0).detach()
        #             # print(action_probs.shape)
        #             # print(action_probs.data.max(0))
        #             tmp_action = action_probs.data.max(0)[1].view(-1, 1)
        #         else: # 全避難所を誘導先候補にする
        #             action_probs = F.softmax(actor_output[i,idxs], dim=0)
        #             tmp_action = action_probs.multinomial(num_samples=1)
        #         action = idxs[tmp_action]
        #         ret[i,0] = action
        #     # print(ret.shape)
        #     return ret
        # else:
        if flg_greedy:
            action_probs = F.softmax(actor_output, dim=1).detach()
            action       = action_probs.data.max(1)[1].view(-1, 1)
        else:
            action_probs = F.softmax(actor_output, dim=1)
            action       = action_probs.multinomial(num_samples=1)
        # print(action.shape)
        return action
 
    def act_greedy(self, x):
        return self.act(x, flg_greedy=True)

        # value, actor_output = self(x)
        # legal_actions = self.legal_actions(x)
        # action_probs = F.softmax(actor_output[legal_actions], dim=1).detach()
        # # print(action_probs)
        # # action       = action_probs.data.max(1)[1].view(1, 1)
        # tmp_action       = action_probs.data.max(1)[1].view(-1, 1)
        # action = legal_actions[tmp_action]
        # # action       = action_probs.data.max(1)[1].view(-1, 1)
        # return action

    def get_value(self, x):
        # return state-value
        value, actor_output = self(x)
        return value

    def evaluate_actions(self, x, actions):
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)

        action_log_probs = log_probs.gather(1, actions)

        probs   = F.softmax(actor_output, dim=1)
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy
