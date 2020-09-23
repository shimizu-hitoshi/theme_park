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
flg_debug = False # True

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

def check_dependency(i,hist,dependency):
    for h in hist:
        if h in dependency.graph[i]: # 後から乗るべきアトラクションに先に乗っていたら
            return False # 選ばない
    return True # OK

def check_dependency2(i,hist,dependency):
    for j in dependency.graph[i]:
        if j not in hist: # 先に乗るべきアトラクションに乗っていなかったら
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
            if not check_dependency2(i,self.hist,dependency): # 制約対象
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
    def __init__(self, ddir, logdir, config):
        self.mode = config['SIMULATION']['mode']
        self.max_iteration = config.getint('SIMULATION', 'max_iteration')
        self.logdir = logdir
        self.ddir = ddir

        # ファイル読み込み
        self.distance_matrix = np.loadtxt("%s/distance.csv"%ddir, delimiter=",")        
        self.attractions = mk_attraction("%s/attraction.csv"%ddir)
        self.guest_json  = json.load(open("%s/guest.json"%ddir))

    def reset(self, dependency):
        self.dependency = dependency
        np.random.seed(0)    #config グループ作成など
        self.nTime = 1           #現在時刻

        self.guests      = {}
        for i, d in enumerate(self.guest_json):
            g   = Guest(d)
            self.guests[g.idx] = g
        self.N = len(self.guests)
        self.LogQueue = LogWriter(self.logdir+"/queue_length.csv")
        self.LogQueue.addHeader(self.attractions)
        self.LogWait = LogWriter(self.logdir+"/wait_time.csv")
        self.LogWait.addHeader(self.attractions)
        self.LogCount = LogWriter(self.logdir+"/attraction_cnt.csv")
        self.LogCount.addHeader(self.attractions)
        self.LogState = LogWriter(self.logdir+"/state.csv")
        self.LogState.addLine(["time", "no enter", "move", "queue", "ride", "wait", "exit"])

        self.born    = self.guests.keys()
        self.active  = []
        self.wait    = []
        self.move    = []
        self.ride    = []
        self.dead    = []
        self.totalWait = 0
        self.totalQueue = 0
        self.totalInpark = 0

    def simulate(self, dependency):
        self.reset(dependency)
        for t in range(1, self.max_iteration+1):
            self.nTime = t
            # print(self.nTime)
            if len(self.dead) == self.N:
                break
            self.iterate()
            self.appendLog()

    def iterate(self):
        self.active.extend([g for g in self.born if self.guests[g].born <= self.nTime ])
        self.born = [g for g in self.born if not self.guests[g].born <= self.nTime ]
    
        #アトラクション更新
        for a in self.attractions:
            self.ride = self.attractions[a].update(self.guests, self.ride, self.max_iteration, self.nTime)
#        print(ride)
    
        #待機状態の人をアクティブに
        for g in self.wait:
            self.guests[g].wait -= 1
            if(self.guests[g].wait <=0):
                self.active.append(g)
        self.wait    = [g for g in self.wait if self.guests[g].wait>0]
        
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

    def saveLog(self):
        self.LogQueue.save()
        self.LogWait.save()
        self.LogCount.save()
        self.LogState.save()        

    def evaluate(self):
        list_surplus = np.array( [ v.surplus for k,v in sorted( self.guests.items() )] )
        mean_surplus = np.mean(list_surplus)
        return mean_surplus


def sim_park(ddir, logdir, config, dependency={}):
    # config = configparser.ConfigParser()
    # config.read(configfn)
    mode = config['SIMULATION']['mode']
    # ddir = config['SIMULATION']['ddir']
    # logdir = config['SIMULATION']['logdir']
    max_iteration = config.getint('SIMULATION', 'max_iteration')
    # mode="normal"
    # dependency={}

    distance_matrix_fn = "%s/distance.csv"%ddir
    distance_matrix = np.loadtxt(distance_matrix_fn, delimiter=",")
    
    #####Configs###########################
    np.random.seed(0)    #config グループ作成など
    # np.random.seed(0)
    global  nTime           #現在時刻

    attractions = mk_attraction("%s/attraction.csv"%ddir)

    guests      = {}
    guestfn = "%s/guest.json"%ddir
    fp  = open(guestfn)
    jsdata  = json.load(fp)
    fp.close()
    for i, d in enumerate(jsdata):
        # g   = Guest(d["idx"])
        g   = Guest(d)
        # for k in d:
        #     g.__dict__[k]   = d[k]
        guests[g.idx] = g

    N = len(guests)
    
    if flg_log:
        fp = {}
        LOGS    = {}
        fp["queue"] = open(logdir+"/queue_length.csv", "w")
        # fp["queue"] = open(logdir+"/queue_length.csv", "w", newline='')
        LOGS["queue"] = csv.writer(fp["queue"])
        line    = ["time"]
        for a in sorted(attractions.keys()):
            line.append("Attraction"+str(a))
        LOGS["queue"].writerow(line)

        fp["wait"] = open(logdir+"/wait_time.csv", "w")
        # fp["wait"] = open(logdir+"/wait_time.csv", "w", newline='')
        LOGS["wait"] = csv.writer(fp["wait"])
        line    = ["time"]
        for a in sorted(attractions.keys()):
            line.append("Attraction"+str(a))
        LOGS["wait"].writerow(line)

        fp["aCnt"] = open(logdir+"/attraction_cnt.csv", "w")
        # fp["aCnt"] = open(logdir+"/attraction_cnt.csv", "w", newline='')
        LOGS["aCnt"] = csv.writer(fp["aCnt"])
        line    = ["time"]
        for a in sorted(attractions.keys()):
            line.append("Attraction"+str(a))
        LOGS["aCnt"].writerow(line)

        fp["state"] = open(logdir+"/state.csv", "w")
        # fp["state"] = open(logdir+"/state.csv", "w", newline='')
        LOGS["state"] = csv.writer(fp["state"])
        line    = ["time", "no enter", "move", "queue", "ride", "wait", "exit"]
        LOGS["state"].writerow(line)
    
    born    = guests.keys()
    active  = []
    wait    = []
    move    = []
    ride    = []
    dead    = []
    totalWait = 0
    totalQueue = 0
    totalInpark = 0

    for nTime in range(1, max_iteration+1):
        if len(dead) == N:
            break
        #入場処理
        active.extend([g for g in born if guests[g].born <= nTime ])
        born = [g for g in born if not guests[g].born <= nTime ]
    
        #アトラクション更新
        for a in attractions:
            ride = attractions[a].update(guests, ride, max_iteration, nTime)
#        print(ride)
    
        #待機状態の人をアクティブに
        for g in wait:
            guests[g].wait -= 1
            if(guests[g].wait <=0):
                active.append(g)
        wait    = [g for g in wait if guests[g].wait>0]
        
        #移動状態の人の処理
        np.random.shuffle(move)
        for g in move:
            guests[g].move -= 1
            if(guests[g].move <=0):
                att = guests[g].dest
                guests[g].pos   = att
                # if guests[g].pos == 0 and len(guests[g].wish)==0:
                # if guests[g].pos == 0 and guests[g].num_wish == guests[g].exp:
                #     guests[g].dead = nTime
                #     dead.append(g)
                #     continue

                # wish達成とtired状態の判定を統合しました20200424
                if guests[g].check_exit(attractions):
                    guests[g].dead = nTime
                    dead.append(g)
                    continue
                ##########DEBUG 全て並ぶ##########
                attractions[att].queue.append(g)
                guests[g].q_start = copy.deepcopy( nTime )
                guests[g].logs.append([nTime, "queueing", att])
                continue
                ##################################
                # #来たけど乗れない
                # if(len(attractions[att].queue)>=attractions[att].remains):
                #     guests[g].wait  =1
                #     wait.append(g)
                #     guests[g].logs.append([nTime, "retire", att])
                # else:
                #     attractions[att].queue.append(g)
                #     guests[g].logs.append([nTime, "queueing", att])
        move    = [g for g in move if guests[g].move>0]
    
    
        #体験中の人の処理
        for g in ride:
            guests[g].ride  -= 1
            if(guests[g].ride <=0):
#                guests[g].wait  =1
#                wait.append(g)
                active.append(g)
    #            
        ride    = [g for g in ride if guests[g].ride>0]
    #
        #ゲストの次のアクション策定(並列処理予定)
#        dead.extend([g for g in active if (guests[g].dead <= nTime or guests[g].exp >= guests[g].num_wish)] )
#        active = [g for g in active if not (guests[g].dead <= nTime or guests[g].exp >= guests[g].num_wish)]
#        dead.extend([g for g in active if (guests[g].exp >= guests[g].num_wish)] )
#        active = [g for g in active if not (guests[g].exp >= guests[g].num_wish)]
        active = [g for g in active if not (guests[g].pos == 0 and guests[g].num_wish==0)]
        # active = [g for g in active if not (guests[g].pos == 0 and len(guests[g].wish)==0)]
    
        for g in active:
            att = guests[g].selAttraction(attractions, mode, dependency)
            # att = selAttraction(guests[g], attractions, mode, dependency)
            # move time from next att
            if att is None:
                tm = None
            else:
                tm = distance_matrix[guests[g].pos, att]
            #乗りたいアトラクションがなければ1〜30分休憩
            if(att == None):
                r = np.random.randint(1, 30+1)
                guests[g].wait  = r
                wait.append(g)
            else:
                guests[g].move  = np.random.randint(tm+1)
                guests[g].dest  = att
                move.append(g)
                guests[g].logs.append([nTime, "move", guests[g].pos, att])
                
        active  = []
        cnt = 0 # 行列の人数
        qlen    = [nTime]
        wTime   = [nTime]
        aCnt = [nTime] # 各アトラクションを体験した人数
        for a in sorted(attractions.keys()):
            cnt+=len(attractions[a].queue)
            qlen.append(len(attractions[a].queue))
            wTime.append(attractions[a].wait)
            aCnt.append(attractions[a].cnt)

        totalWait += len(wait)
        totalQueue += cnt
        totalInpark += len(move) + cnt + len(ride) + len(wait) # 

        if flg_log:
            LOGS["queue"].writerow(qlen)
            LOGS["wait"].writerow(wTime)
            LOGS["aCnt"].writerow(aCnt)

            line    = [nTime, len(born), len(move),cnt, len(ride), len(wait), len(dead)]
            LOGS["state"].writerow(line)
        if flg_debug : print( nTime, "no born", len(born), len(active), "wait", len(wait), "exit",len(dead), "queue", cnt, "move", len(move), "ride", len(ride) )
    
    if flg_log:
        for k, _fp in fp.items():
            _fp.close()
    
# 実行時間短縮のために，グラフ保存を抑制
    # ifn = "%s/wait_time.csv"%logdir
    # ofn = "%s/wait_time.png"%logdir
    # plotter.plot_data_n(ofn, ifn, N, tick_interval=1)

    # ifn = "%s/state.csv"%logdir
    # ofn = "%s/state.png"%logdir
    # plotter.cum_plot(ofn, ifn, tick_interval=1)

    shutil.copy2(__file__, logdir)
    
#     来園者の履歴を画面表示
    list_exp = np.array( [ v.exp for k,v in sorted( guests.items() )] )
    list_util = np.array( [ v.util for k,v in sorted( guests.items() )] )
    list_surplus = np.array( [ v.surplus for k,v in sorted( guests.items() )] )



#     list_exp = []
#     list_util = []
#     # total_util = 0
# #    list_wait = []

#     for g in sorted(guests.keys()):
#         list_exp.append( guests[g].exp )
#         list_util.append( guests[g].util )
#         # total_util += guests[g].util
# #        print(guests[g].exp)
# #    ofn = "%s/hist_exp.png"%logdir
# #    plotter.hist_data(ofn,list_exp)

    np.savetxt("%s/list_exp.txt"%logdir, list_exp ,fmt="%d")
    np.savetxt("%s/list_util.txt"%logdir, list_util )

# 実行時間短縮のために，詳細ログ出力を抑制
    if flg_log:
        ofn = "%s/result.json"%logdir
        saveResult(ofn, guests)
#    ana_sim.ana_sim(logdir)

    total_exp = np.sum(list_exp)
    mean_exp = np.mean(list_exp) # 20190705 result.json集計と比べて過大，バグかな？
    ret_queue = 1.0 * totalQueue / N
    mean_wait = 1.0 * totalWait / N
    mean_Inpark = 1.0 * totalInpark / N
    mean_util = np.mean(list_util)
    mean_surplus = np.mean(list_surplus)
    # mean_util = total_util / N
#    print("average of # exp : %s"%np.mean(list_exp))
#    print("average of wait : %s"%(totalWait / N))
    return total_exp, mean_exp, ret_queue, mean_wait, mean_Inpark, nTime, mean_util, mean_surplus

if __name__ == '__main__':
    ddir = "data"
    datehead = datetime.today().strftime("%Y%m%d_%H%M%S")
    logdir  = "results/%s"%datehead
    if(not os.path.exists(logdir)):
    #    os.mkdir(logdir, 0755)
        os.mkdir(logdir)

    sim_park(ddir, logdir, [0] * 5)

