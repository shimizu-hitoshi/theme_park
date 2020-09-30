#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import gym
import gym.spaces
from gym.utils import seeding
import numpy as np
import configparser
sys.path.append('../')
# from controler import FixControler
import copy
from myenv.sim_park import SimPark

DEBUG = True # False # True # False # True # False

class SimEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super().__init__()
        self.reward_range = [-1., 1.]

    def step(self, action): # これを呼ぶときには，actionは決定されている
        # self.state = self._get_state() # 1stepにつき冒頭と末尾に状態取得？
        # print(action)
        # print(action.shape)
        # sys.exit()

        # if len(self.agents) == 1: # エージェント数が1だとスカラーになってエラー->暫定対処
        #     action = action.reshape(-1,1)

        _action = self._get_action(action)
        self.park.set_restriction(_action)
        self.park.step(self.interval)
        # self.call_traffic_regulation(_action, self.num_step)
        # print("dict_actions",dict_actions)
        # self.call_traffic_regulation(dict_actions, self.num_step)
        # self.call_iterate(self.cur_time + self.interval) # iterate
        self.cur_time += self.interval
        # self.update_navi_state() # self.navi_stateを更新するだけ
        self.state = self._get_state()
        # self.state = self._get_observation(self.cur_time + self.interval) # iterate
        # observation = self.state2obsv( self.state, self.id ) 
        # observation = self.state 
        # reward = self._get_reward_time()
        # reward = self._get_reward()
        # reward = self._get_reward(self.edge_state)
        # sum_pop = np.sum(self.edge_state) * self.interval / self.num_agents # 累積すると平均移動時間

        reward = self._get_reward()
        print("reward",reward)
        # self.episode_reward += sum_pop
        self.episode_reward += reward
        # print("CURRENT", self.cur_time, action, sum_pop, self.T_open[self.num_step], reward, self.episode_reward)
        # if DEBUG: print("CURRENT", self.env_id, self.cur_time, action, sum_pop, reward, self.episode_reward)
        with open(self.resdir + "/current_log.txt", "a") as f:
            # f.write("CURRENT {:} {:} {:} {:} {:} {:}\n".format(self.env_id, self.cur_time, action, sum_pop, reward, self.episode_reward))
            f.write("CURRENT {:} {:} {:} {:} {:}\n".format(self.env_id, self.cur_time, action, reward, self.episode_reward))
        self.num_step += 1
        done = self.max_step <= self.num_step
        # travel_time = self.mk_travel_open()
        info = {}
        if done:
            # agentid, travel_time = self._goal_time_all() # 歩行者の移動速度リストを取得
            info = {
                    "episode": {
                        "r": self.episode_reward
                        },
                    "events": self.event_history,
                    "env_id":self.env_id,
                    # "travel_time":travel_time,
                    # "agentid":agentid
                    }
            # print("info",info)
        if DEBUG: print(self.state.shape, reward, done, info)
        return self.state, reward, done, info # obseration, reward, done, info

    def reset(self):
        # config = configparser.ConfigParser()
        # config.read('config.ini')
        if DEBUG: print("reset")
        self.sim_time  = self.config.getint('SIMULATION', 'sim_time')
        self.interval  = self.config.getint('SIMULATION', 'interval')
        self.max_step  = int( np.ceil( self.sim_time / self.interval ))
        # self.max_step = self.sim_time
        self.cur_time  = 0
        self.num_step  = 0
        self.state     = np.zeros(self.num_obsv * self.obs_step)

        self.episode_reward = 0
        self.event_history = []
        # self.flag = True

        # for reward selection
        # self.prev_goal = 0

        # self.reset_sim() # set up simulation
        self.park.reset()
        return self.state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None, env_id=None, datadirs=None, config=None, R_base=(None, None)):
        self.park = SimPark(datadirs, config)

        print(R_base)
        self.T_open, self.travel_open = R_base
        # print("T_open @ seed",self.T_open)
        # print("travel_open @ seed",self.travel_open)
        # training_targets = dict_target["training"]
        # fixed_agents = dict_target["fixed"] # その他を固定しよう
        # rule_agents = dict_target["rule"]
        # fixed_agents: モデルで行動，更新なし，の避難所
        # training_targets: 学習対象の避難所
        # rule_agents: ルールベースの避難所

        # from init (for config import)
        self.config = config
        # num_parallel   = config.getint('TRAINING',   'num_parallel')
        # tmp_id = len(training_targets) % num_parallel
        # tmp_id = seed % len(training_targets)
        # tmp_id = env_id % len(datadirs)
        # if DEBUG: print(training_targets, tmp_id)
        self.env_id = env_id
        # self.sid = training_targets[tmp_id]
        # self.training_target = self.sid # 不要かも
        # self.datadir = datadirs[tmp_id]
        self.datadir = datadirs[0]
        # config = configparser.ConfigParser()
        # config.read('config.ini')
        # self.num_agents = config.getint('SIMULATION', 'num_agents')
        # self.num_edges  = config.getint('SIMULATION', 'num_edges')
        self.obs_step   = config.getint('TRAINING',   'obs_step')
        # self.obs_degree   = config.getint('TRAINING',   'obs_degree')
        # self.datadir         = config.get('SIMULATION',    'datadir')
        self.tmp_resdir = config['TRAINING']['resdir']
        self.park.set_logdir(self.tmp_resdir )
        self.actions = np.loadtxt( config['SIMULATION']['actionfn'] , dtype=int )
        # self.agents = training_targets # = self.actions
        # self.agents = copy.deepcopy(self.actions)
        # if DEBUG: print(self.actions)
        # sys.exit()
        # self.dict_action = {}
        # for action in list( self.actions ):
        #     self.dict_action[]
        # self.flg_reward = config['TRAINING']['flg_reward']

        # self.flag = True
        self.num_attractions = 5
        # self.actions = np.arange(1,self.num_attractions+1,1) # [1,2,3,4,5]
        # self.actions = list(range(1,self.num_attractions+1,1)) # [1,2,3,4,5]
        # self.edges = Edge(self.obs_degree) # degreeは不要になったはず．．．
        # self.edges = Edge(self.datadir) # degreeは不要になったはず．．．
        # ->seed()の前に設定してしまいたい
        # self.num_edges = self.edges.num_obsv_edge
        # self.num_goals = self.edges.num_obsv_goal
        # self.num_navi = len(self.actions) * len(self.actions) # 誘導の状態数は，ワンホットベクトルを想定
        # self.navi_state = np.zeros(len(self.actions) * len(self.actions), dtype=float) # 入れ物だけ作っておく
        # self.navi_state = np.zeros((3* 3), dtype=float) # 入れ物だけ作っておく
        # self.num_navi = 3 * 3 # 誘導の状態数は，ワンホットベクトルを想定
        # self.navi_state = np.zeros(len(self.actions) * len(self.agents), dtype=float) # 入れ物だけ作っておく
        # self.num_obsv = self.num_edges + self.num_goals # １ステップ分の観測の数
        # if DEBUG: print("self.navi_state.shape", self.navi_state.shape)
        # self.num_obsv = self.num_edges + self.num_goals + self.num_navi # １ステップ分の観測の数
        # self.num_obsv = self.num_attractions + self.num_navi
        self.num_obsv = self.num_attractions + 7

        self.action_space      = gym.spaces.Discrete(self.actions.shape[0])
        self.observation_space = gym.spaces.Box(
                low=0,
                high=100000,
                # high=self.num_agents,
                shape=np.zeros(self.num_obsv * self.obs_step).shape
                )
        # assert self.action_space.n == self.actions.shape[0]
        assert self.observation_space.shape[0] == self.num_obsv * self.obs_step

        # self.state = None
        # self.state     = np.zeros(self.num_edges * self.obs_step)
        # self.cur_time  = 0
        # self.interval 
        # self.prev_goal = 0

        # self.reset()
        # copy from reset()
        self.sim_time  = self.config.getint('SIMULATION', 'sim_time')
        # self.interval  = self.config.getint('SIMULATION', 'interval')
        # self.max_step  = int( np.ceil( self.sim_time / self.interval ))
        self.max_step = self.sim_time
        self.cur_time  = 0
        self.num_step  = 0
        self.state     = np.zeros(self.num_obsv * self.obs_step)

        # original seed
        # self.np_random, seed = seeding.np_random(seed)
        # https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
        seeding.np_random(seed) 
        # self.set_datadir(self.datadir)
        # print(self.datadir)
        self.set_resdir("%s/sim_result_%d"%(self.tmp_resdir, self.env_id))
        # ルールベースの避難所のエージェントを生成する
        # self.others = {}
        # for shelter_id, node_id in enumerate( self.actions ):
        #     # 自分のエージェントを作ってもいいけど，使わない
        #     controler = FixControler(shelter_id, self.edges.DistanceMatrix)
        #     self.others[shelter_id] = controler

        return [seed]

    def _get_action(self, action):
        # print("get_action")
        # print(self.actions)
        # print(action)
        return self.actions[action,:]

    def _get_reward(self):# based on surplus
        print("max_step", self.max_step, "num_step", self.num_step)
        # if self.travel_open is None:
        #     return 0
        if self.max_step > self.num_step+1:
            return 0 # reward only last step
        else:
            surplus = self.park.evaluate()
            reward = (surplus - self.travel_open) / self.travel_open
            return reward # 上限を1にしなくてもよいかも
        # agentid, travel_time = self._goal_time_all()
        # # print(agentid, travel_time)
        # # if len(agentid) == 0:
        # #     return 0
        # if len(agentid) != self.num_agents:
        #     return -1
        # reward = np.sum( self.travel_open[agentid] - travel_time ) / np.sum( self.travel_open[agentid] )
        # # reward = np.sum( self.T_open[agentid] - travel_time ) / np.sum( self.T_open[agentid] )
        # if reward < 0:
        #     return max(reward, -1)
        # return min(reward, 1)



    # def _edge_cnt(self):
    #     ret   = np.zeros(len(self.edges.observed_edge))
    #     for e, idx in self.edges.observed_edge.items():
    #         fr, to     = e
    #         ret[idx]   = self.lib.cntOnEdge(fr-1, to-1)
    #     return ret

    # def _goal_cnt(self): # ゴールした人の累積人数
    #     ret   = np.zeros(len(self.edges.observed_goal)) # 返値は，observed_goal次元
    #     stop_time = self.cur_time # + self.interval
    #     start_time = 0 # self.cur_time #stop_time - self.interval
    #     for node, idx in sorted( self.edges.observed_goal.items() ):
    #         tmp = self.lib.goalAgentCnt(start_time, stop_time-1, node-1)
    #         ret[idx]   = tmp
    #         # print(start_time, stop_time, node-1, idx, tmp)
    #     return ret


    # def update_navi_state(self): # この関数，本当に必要？
    #     self.navi_state = self.tmp_action_matrix.flatten()

    def _get_state(self): # 時刻は進めない->繰り返し使うと同じstateが最大4step分だけ繰り返されるので注意
        # １ステップ分ずらす
        obs     = self.state[self.num_obsv:] # 左端を捨てる
        # 避難所の状況を取得
        # tmp_goal_state = copy.deepcopy( self._goal_cnt() )
        # print(tmp_goal_state)
        # self.edge_state = copy.deepcopy( self._edge_cnt() ) # 何度も使いそうなので保存
        # self.goal_state = self.edges.goal_capa - tmp_goal_state # 何度も使いそうなので保存
        # print(self.goal_state)
        # print(np.sum(tmp_goal_state), np.sum(self.edge_state))
        # cur_obs = np.append(self.edge_state , self.goal_state )
        # cur_obs = np.append(cur_obs , self.navi_state )
        cur_obs = self.park._get_state()
        print("cur_obs",cur_obs)
        return np.append(obs, cur_obs) # 右端に追加

    # def set_datadir(self, datadir):
    #     self.datadir = datadir
    #     agentfn    = os.path.dirname(os.path.abspath(__file__)) + "/../%s/agentlist.txt"%self.datadir
    #     # self.speed = self.get_speed(agentfn)
    #     self.num_agents = self.get_num_agents(agentfn)

    def set_resdir(self, resdir):
        # print(resdir)
        os.makedirs(resdir, exist_ok=True)
        self.resdir = resdir

