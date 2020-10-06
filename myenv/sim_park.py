#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys, os, re, copy
import json
import numpy as np
import csv
from datetime import datetime
import shutil
#import argparse
# import ana_sim
# import plotter
import configparser
from collections import defaultdict
import random

flg_log = True # False
# flg_log = False # False # False
# DEBUG = False # True
DEBUG = False # True # False # True

eps = 10e-6

def rand_range(a,b):
    # a から b までの実数一様分布
    ret = (b - a) * np.random.rand() + a
    return ret

def rand_normal(mu, sigma):
    # sigmaが0ならmuを返す正規分布
    # sizeはとりあえず無視してスカラーのみ対応
    if sigma == 0:
        ret = mu
    else:
        ret = np.random.normal(mu,sigma)
    return ret

# def check_dependency(i,hist,dependency):
#     for h in hist:
#         # if h in dependency.graph[i]: # 後から乗るべきアトラクションに先に乗っていたら
#         if h in dependency[i]: # ただのdict->listに変更
#             return False # 選ばない
#     return True # OK

# def check_dependency2(i,hist,dependency):
#     for j in dependency.graph[i]:
#         if j not in hist: # 先に乗るべきアトラクションに乗っていなかったら
#             return False # 選ばない
#     return True # OK

def check_dependency3(i,hist,dependency):
    if i in dependency: # ただのlistに変更
        return False # 選ばない
    return True # OK


class   Attraction:
    # remains = 0         #閉園までの残数
    def var_dump(self):
        return self.__dict__

    def __init__(self, idx, st, capa):

        self.idx = idx
        self.queue  = []

        self.capa    = int(capa)
        self.duration   = int(st)
        self.span   = st
        self.throughput = 1.0 * self.capa / self.span

        self.idx = idx
        self.queue  = []
        # self.remains = 1.0 * max_iteration * self.throughput
        self.remains = 0
        self.wait    = 0         #現在の待ち時間
        self.cnt     = 0         #累積搭乗人数
        self.wait_display = 0    #待ち時間表示
        # self.max_iteration = max_iteration

    def update(self, guests, ride, max_iteration, nTime):
        #稼働
        if(nTime % self.span==0):
            for n in range(self.capa):
                if(len(self.queue)==0):
                    break
                #待機列から除外
                g       = self.queue.pop(0)
                #アトラクション体験時間を設定
                guests[g].ride  = self.duration
                ride.append(g)

                guests[g].exp   += 1 # 体験回数：退園判定に使う
                # guests[g].util  += guests[g].alpha[str(self.idx)] # 許容限界減衰モデルでは使えない
                guests[g].util  += guests[g].alpha_hat[str(self.idx)] # 許容限界減衰モデルに対応
                _wait_time = nTime - guests[g].q_start # 実際に待った時間
                guests[g].surplus  += guests[g].alpha_hat[str(self.idx)] - _wait_time # 待ち時間を差し引く
                guests[g].hist.append(guests[g].pos)
                # ログに体験時点のalpha_hatを追加: 2020/04/27
                guests[g].logs.append([nTime, "ride", self.idx, guests[g].alpha_hat[str(self.idx)]])
                guests[g].L[str(self.idx)] += 1
                guests[g].update_alpha()
#                guests[g].pay   += self.price
                self.cnt        += 1
                # self.remains -= 1 # 残席を１個減らす -> 空いてるときに不具合

        # 残数と待ち時間の更新
        self.remains    = 1.0 * (max_iteration - nTime) * self.throughput
#        self.remains    = ((max_iteration-self.duration-1) / self.span - nTime /self.span) * self.capa
        # self.wait    = (len(self.queue) / self.num + 1)*self.span - nTime % self.span
        self.wait    = 1.0 * len(self.queue) / self.throughput
        self.wait_display = self.wait# + self.margin

        return ride

class   Guest:
    def var_dump(self):
        return self.__dict__
    # def __init__(self, idx):
    def __init__(self, d):
        for k in d:
            # print(k, d[k])
            self.__dict__[k]   = d[k]

        M = len(self.__dict__["alpha"])
        self.idx = self.__dict__["idx"]
        self.logs   = []
        self.wait = 0 # 残り待機時間
        self.move = 0 # 残り移動時間
        self.exp = 0 # 体験個数
        # edit 20190620
        self.pos  = 0
        self.dest = 0
        self.dead = -1
        self.hist = []
        self.util = 0 # 待ち時間を差し引く前
        self.surplus = 0 # 待ち時間を差し引く後
        self.q_start = -1 # 待ち行列に並んだ時刻を一時的に保持
        self.L = {}
        for m in range(1,M+1,1):
            self.L[str(m)] = 0
        self.alpha_hat = {}
        self.alpha_hat = copy.deepcopy(self.alpha)
        # for m in range(1,M+1,1):
        #     self.alpha_hat[str(m)] = 0
        self.tired = False
    def check_exit(self, attractions):
        if self.pos == 0 and self.num_wish == self.exp:
            return True
        if self.pos == 0 and self.tired:
            return True
        # print(self.__dict__)
        return False
    def update_alpha(self):
        for m in self.alpha_hat:
            self.alpha_hat[m] = max(0, self.alpha[m] * (1 - self.L[m] / (self.Lhat[m] + eps)) )

    # def selAttraction(g, attractions, distance_matrix, mode="linear", dependency={}):
    def selAttraction(self, attractions, mode="linear", dependency={}):
        # 許容限界モデル = "linear" に限定
        if self.num_wish == self.exp: # 達成
            return 0 # exit
        self.tired = True # 体験回数が増えてどれも乗りたいアトラクションがないなら退園
        prob = []
        for i in attractions.keys():
            idx = str(i)
            if self.alpha_hat[idx] > 0:
                self.tired = False
            # if not check_dependency(i,self.hist,dependency): # 制約対象
            # if not check_dependency2(i,self.hist,dependency): # 制約対象
            if not check_dependency3(i,self.hist,dependency): # 制約対象
                p = 0
                # break
            elif attractions[i].remains <= len(attractions[i].queue): # 残席なし
                p = 0
            else:
                if mode == "linear" or mode == "maxGS":
                    p   = max(0,  self.alpha_hat[idx] - attractions[i].wait)
                elif mode == "minWT":
                    if self.alpha_hat[idx] - attractions[i].wait < 0:
                        p = np.inf
                    else:
                        p = attractions[i].wait
                elif mode == "maxTL":
                    if self.alpha_hat[idx] - attractions[i].wait < 0:
                        p = 0
                    else:
                        p  = self.alpha_hat[idx]
                else: # 想定外で未実装
                    pass

            prob.append(p)
        if self.tired:
            return 0

        if mode == "linear":
            return self.selLinear(prob)
        elif mode == "minWT":
            return self.selminProb(prob)
        elif mode == "maxTL" or mode == "maxGS":
            return self.selmaxProb(prob)
        else: # mode == "maxTL"
            pass

    def selLinear(self, prob):
        prob = np.array(prob)
        if np.sum(prob) == 0: # 全て許容不能
            return None
        prob = 1.0 * prob / np.sum(prob)
        i = np.argmax(np.random.multinomial(1,prob))+1
        if i == 0:
            print("error: prob all zero") # for debug
        return i

    def selminProb(self, prob):
        if np.min(prob) == np.inf: # 全て許容不能
            return None
        i = np.argmin(prob) +1 # 複数のアトラクションで待ち時間0なら，たぶんIDの小さいほうが選ばれる？
        return i

    def selmaxProb(self, prob):
        if np.sum(prob) == 0: # 全て許容不能
            return None
        i = np.argmax(prob) +1 # 複数のアトラクションでprobが最小なら，たぶんIDの小さいほうが選ばれる？
        return i


# https://wtnvenga.hatenablog.com/entry/2018/05/27/113848
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def saveResult(fn, guests):
    # jsdata  = {}
    jsdata  = []
#    for a in sorted(attractions.keys()):
#        jsdata.append(attractions[a].var_dump())
    for g in sorted(guests.keys()):
        tmp = guests[g].var_dump()
        jsdata.append( tmp )
        # idx = tmp["idx"]
        # jsdata[str(idx)] = tmp
    # print(jsdata["1"])
    fp  = open(fn, "w")
    tmp2 = json.dumps(jsdata, indent=2, cls = MyEncoder)
    # for i in range(1,1000+1,1):
    #     # print(i,jsdata[str(i)])
    #     tmp2 = json.dumps(jsdata[str(i)], indent=2, cls = MyEncoder)
    #     # print(i,tmp2)
    fp.write(tmp2)
    fp.close()

def mk_attraction(fn):
    # attraction = np.loadtxt("attraction.csv", delimiter=",")
    attraction = np.loadtxt(fn, delimiter=",")
    M = attraction.shape[0]
    attractions = {}
    for i in range(1, M+1):
        # a   = Attraction(i, st=attraction[i-1,0], capa=attraction[i-1,1])
        a   = Attraction(i, attraction[i-1,0], attraction[i-1,1])
        attractions[i]=a
    return attractions

class LogWriter:
    def __init__(self, fn):
        self.log = []
        self.fn = fn

    def addHeader(self, attractions):
        line    = ["time"]
        for a in sorted(attractions.keys()):
            line.append("Attraction"+str(a))
        self.log.append(line) 

    def addLine(self, line):
        self.log.append(line) 

    def save(self):
        with open(self.fn, "w") as f:
            w = csv.writer(f)
            for line in self.log:
                w.writerow(line)

class SimPark:
    # def __init__(self, ddir, logdir, config):
    def __init__(self, ddir, config):
        self.mode = config['SIMULATION']['mode']
        self.max_iteration = config.getint('SIMULATION', 'max_iteration')
        # self.logdir = logdir
        self.ddir = ddir

        # ファイル読み込み
        self.distance_matrix = np.loadtxt("%s/distance.csv"%ddir, delimiter=",")        
        self.attractions = mk_attraction("%s/attraction.csv"%ddir)
        self.guest_json  = json.load(open("%s/guest.json"%ddir))

    def set_logdir(self, logdir):
        self.logdir = logdir

    # def reset(self, dependency):
    def reset(self):
        self.dependency = {}
        self.enterance_restriction = False # 入園規制フラグ<-set_restrictionで設定
        # self.dependency = dependency
        np.random.seed(0)    #config グループ作成など
        self.nTime = 0 # 1           #現在時刻

        self.guests      = {}
        for i, d in enumerate(self.guest_json):
            g   = Guest(d)
            self.guests[g.idx] = g
        self.N = len(self.guests)
        self.M = len(self.attractions)
        self.LogQueue = LogWriter(self.logdir+"/queue_length.csv")
        self.LogQueue.addHeader(self.attractions)
        self.LogWait = LogWriter(self.logdir+"/wait_time.csv")
        self.LogWait.addHeader(self.attractions)
        self.LogCount = LogWriter(self.logdir+"/attraction_cnt.csv")
        self.LogCount.addHeader(self.attractions)
        self.LogState = LogWriter(self.logdir+"/state.csv")
        self.LogState.addLine(["time", "no enter", "move", "queue", "ride", "wait", "exit"])
        self.LogRestriction = LogWriter(self.logdir+"/restriction.csv")
        self.LogState.addLine(["att1","att2","att3","att4","att5","ent"])

        self.born    = self.guests.keys()
        self.active  = []
        self.wait    = []
        self.move    = []
        self.ride    = []
        self.dead    = []
        self.totalWait = 0
        self.totalQueue = 0
        self.totalInpark = 0

    # def simulate(self):
    # # def simulate(self, dependency):
    #     """
    #     最初から最後までシミュレーションを実行する
    #     """
    #     # self.reset(dependency)
    #     self.reset()
    #     for t in range(1, self.max_iteration+1):
    #         self.nTime = t
    #         # print(self.nTime)
    #         if len(self.dead) == self.N:
    #             break
    #         self.iterate()
    #         self.appendLog()

    def set_restriction(self, action):
        """
        後で単体テスト
        """
        self.restriction = action # 6次元のベクトル
        self.dependency = [] # 各アトラクションの受付停止
        # self.action = action
        for i, a in enumerate(action[:-1]):
            if a == 1:
                self.dependency.append(i+1) # アトラクションIDは1はじまり
        # 入園規制
        self.enterance_restriction = (action[-1] == 1) # True or False
        return None

    def step(self, interval):
        for t in range(interval):
            self.iterate()
            self.appendLog() # ログ出力しないならコメントアウトすること
            self.nTime += 1
        return None

    def iterate(self):
        if DEBUG: print("iterate")
        if not self.enterance_restriction: # 入場規制がないときだけ，bornからactiveに移す
            self.active.extend([g for g in self.born if self.guests[g].born <= self.nTime ])
            self.born = [g for g in self.born if not self.guests[g].born <= self.nTime ]
        if DEBUG: print(self.born)
        #アトラクション更新
        for a in self.attractions:
            self.ride = self.attractions[a].update(self.guests, self.ride, self.max_iteration, self.nTime)
        if DEBUG: print(self.ride)
    
        #待機状態の人をアクティブに
        for g in self.wait:
            self.guests[g].wait -= 1
            if(self.guests[g].wait <=0):
                self.active.append(g)
        self.wait    = [g for g in self.wait if self.guests[g].wait>0]
        if DEBUG: print(self.wait)        
        #移動状態の人の処理
        # random.shuffle(self.move)
        # np.random.shuffle(self.move)
        for g in self.move:
            self.guests[g].move -= 1
            if(self.guests[g].move <=0):
                att = copy.deepcopy( self.guests[g].dest )
                self.guests[g].pos   = copy.deepcopy( att )

                # wish達成とtired状態の判定を統合しました20200424
                # if self.guests[g].check_exit(self.attractions):
                if att == 0:# エラー回避のため，出口(att==0)に着いたらチェックなしで退園
                    self.guests[g].dead = self.nTime
                    self.dead.append(g)
                    continue
                ##########DEBUG 全て並ぶ##########
                if att == 0:
                    print(att, self.guests[g].__dict__) # optunaで並列化するとこのあたりでエラー
                    print(att, self.guests[g].hist) # optunaで並列化するとこのあたりでエラー
                    print(att, self.guests[g].logs) # optunaで並列化するとこのあたりでエラー
                self.attractions[att].queue.append(g)
                self.guests[g].q_start = copy.deepcopy( self.nTime )
                self.guests[g].logs.append([self.nTime, "queueing", att])
                continue
                ##################################
        self.move    = [g for g in self.move if self.guests[g].move>0]
    
    
        #体験中の人の処理
        for g in self.ride:
            self.guests[g].ride  -= 1
            if(self.guests[g].ride <=0):
                self.active.append(g)

        self.ride    = [g for g in self.ride if self.guests[g].ride>0]
        self.active = [g for g in self.active if not (self.guests[g].pos == 0 and self.guests[g].num_wish==0)]
    
        for g in self.active:
            att = copy.deepcopy( self.guests[g].selAttraction(self.attractions, self.mode, self.dependency) )
            if att is None:
                tm = None
            else:
                # print(g, att) # for debug
                tm = copy.deepcopy( self.distance_matrix[self.guests[g].pos, att] )
            #乗りたいアトラクションがなければ1〜30分休憩
            if(att == None):
                r = np.random.randint(1, 30+1)
                self.guests[g].wait  = r
                self.wait.append(g)
            else:
                self.guests[g].move  = np.random.randint(tm+1)
                self.guests[g].dest  = att
                self.move.append(g)
                self.guests[g].logs.append([self.nTime, "move", self.guests[g].pos, att])
                
        self.active  = [] # 本来，iterateの冒頭に書くべきでは？

    def _get_state(self):
        ret = []
        cnt = 0 # 行列の人数
        for a in sorted(self.attractions.keys()):
            cnt+=len(self.attractions[a].queue)
            ret.append(len(self.attractions[a].queue))
        ret.extend([self.nTime, len(self.born), len(self.move),cnt, len(self.ride), len(self.wait), len(self.dead)])
        return np.array(ret)

    def appendLog(self):
        # 以下，ログ出力のための処理
        cnt = 0 # 行列の人数
        qlen    = [self.nTime]
        wTime   = [self.nTime]
        aCnt = [self.nTime] # 各アトラクションを体験した人数
        for a in sorted(self.attractions.keys()):
            cnt+=len(self.attractions[a].queue)
            qlen.append(len(self.attractions[a].queue))
            wTime.append(self.attractions[a].wait)
            aCnt.append(self.attractions[a].cnt)

        self.totalWait += len(self.wait)
        self.totalQueue += cnt
        self.totalInpark += len(self.move) + cnt + len(self.ride) + len(self.wait) # 

        self.LogQueue.addLine(qlen)
        self.LogWait.addLine(wTime)
        self.LogCount.addLine(aCnt)
        self.LogState.addLine([self.nTime, len(self.born), len(self.move),cnt, len(self.ride), len(self.wait), len(self.dead)])
        # print( self.nTime, "no born", len(self.born), len(self.active), "wait", len(self.wait), "exit",len(self.dead), "queue", cnt, "move", len(self.move), "ride", len(self.ride) )
        self.LogRestriction.addLine(self.restriction)

    def saveLog(self):
        self.LogQueue.save()
        self.LogWait.save()
        self.LogCount.save()
        self.LogState.save()
        self.LogRestriction.save()

    def evaluate(self):
        list_surplus = np.array( [ v.surplus for k,v in sorted( self.guests.items() )] )
        if DEBUG: print(list_surplus)
        mean_surplus = np.mean(list_surplus)
        return mean_surplus

if __name__ == '__main__':
    park = SimPark(ddir, config)
    ddir = "data"
    datehead = datetime.today().strftime("%Y%m%d_%H%M%S")
    logdir  = "results/%s"%datehead
    if(not os.path.exists(logdir)):
    #    os.mkdir(logdir, 0755)
        os.mkdir(logdir)

    # sim_park(ddir, logdir, [0] * 5)

