#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, re, random
import json
import numpy as np
import csv
from datetime import datetime

import shutil
import argparse
import configparser
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

#import plotter

eps = 10e-6

def topological_sort(wish, D):
#    https://inzkyk.github.io/algorithms/depth_first_search/topological_sort/
# Node status: New -> Active -> Finished
    def visit(w):
        done.append(w)
#        if w in D:
        for d in D.graph[w]:
            if d in wish and d not in done:# New
                visit(d)
            elif d in done and d not in L: # Active
                print("error:cyclic")
                return False
#                    sys.exit()
        L.insert(0,w)
    L = [] # Finished
    done = [] # Active
    for w in wish[::-1]:
        if w in L or w in done: # New
            continue
        visit(w)
    return L

def rand_multinomial(a):
    _tmp_alpha = normalize_simplex(a)
    tmp_dirichlet = np.random.multinomial(1, _tmp_alpha,1)
#    print(tmp_dirichlet)
    tmp_dirichlet = tmp_dirichlet[0]
    return tmp_dirichlet

def normalize_simplex(a):
    tmp_a = np.array(a)
    return tmp_a / np.sum(a)

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

# class   Guest:
#     def var_dump(self):
#         return self.__dict__
#     def __init__(self, idx, attractions, num_wish, N, lam=False):
#         self.idx = idx
#         M = len(attractions)
#         if not lam: # lam is False
#             self.born = int( 1.0 * idx / N * 10000)
#         else:
#             self.born = int( 1.0 * idx / lam)
#         self.wish = random.sample(range(1,M+1,1),num_wish)
#         self.num_wish = num_wish
#         # self.num_wish = len(self.wish)
#         # self.wish = topological_sort(self.wish, dependency)

class   Guest_linear: # 許容限界モデルの来園者
    def var_dump(self):
        return self.__dict__
    def __init__(self, idx, num_wish, N):
    # def __init__(self, idx, attractions, num_wish, N, dependency={}, lam=False):
        self.idx = idx
        # M = len(attractions)
        self.born = int( 1.0 * idx / N * 10000)
        # if not lam: # lam is False
        #     self.born = int( 1.0 * idx / N * 10000)
        # else:
        #     self.born = int( 1.0 * idx / lam)

        # mode = "linear"の場合は，wishを事前に決めない
        # self.wish = random.sample(range(1,M+1,1),num_wish)
        self.num_wish = num_wish
        # self.num_wish = len(self.wish)
        # self.wish = topological_sort(self.wish, dependency)
    def set_alpha(self, config):
        self.alpha = {}
        beta = str2list_float( config["GUEST"]["beta"] )
        mu = float( config["GUEST"]["mu"] )
        sigma = float( config["GUEST"]["sigma"] )
        q = str2list_float( config["GUEST"]["q"] )
        # print( beta, mu, sigma, q)
        # guest_feature = ( beta, mu, sigma, q )
        # beta, mu, sigma, q= guest_feature
        M = len(beta) # num_attraction
        tmp_beta = []
        for m in range(M): # 確率qで選択候補から除外
            if np.random.random() < q[m]:
                tmp_beta.append(beta[m])
            else:
                tmp_beta.append(eps)
        # print(tmp_beta)
        tmp_psi = np.random.dirichlet(tmp_beta)
        tmp_phi = np.random.lognormal(mu,sigma)
        # while True:
        #     tmp_phi = np.random.lognormal(mu,sigma)
        #     if tmp_phi > 1000: # 移動時間の最大値を超えないと，動けない人が生じる
        #         break
        tmp_alpha = tmp_phi * tmp_psi / np.max(tmp_psi)
        for m in range(1,M+1,1):
            self.alpha[m] = tmp_alpha[m-1]

    def set_Lhat(self, config):
        self.Lhat = {}
        avg = str2list_float( config["GUEST"]["avg"] )
        M = len(avg) # num_attraction
        for m in range(1,M+1,1):
            self.Lhat[m] = zero_truncated_poisson(avg[m-1])
            # self.Lhat[m] = tmp_alpha[m-1]


class   Guest_beta(Guest_linear): # 許容限界モデルの来園者 # 入園時刻２ピーク
    def __init__(self, idx, num_wish, N):
        self.idx = idx
        # self.born = self.set_born()
        self.num_wish = num_wish

    def set_born(self, max_iteration):
        #入場時刻決め
        beta_enter1 = [1,10]
        beta_enter2 = [8,12]
        beta = [beta_enter1, beta_enter2]
        # max_iteration   = 780
        if np.random.random() > 0.5:
            _beta_enter = beta[0]
        else:
            _beta_enter = beta[1]
        born   = int( max_iteration * np.random.beta(_beta_enter[0], _beta_enter[1]))
        self.born = born

class Guest_copy(Guest_beta): # アンケートを人数分だけ複製する場合
    def set_alpha_Lhat(self, alphas, rides):
        self.alpha = {}
        self.Lhat = {}
        M = alphas.shape[1] # num_attraction
        N = alphas.shape[0] # num_questionnaire
        n = np.random.randint(0,N)
        for m in range(1,M+1,1):
            self.alpha[m] = alphas[n,m-1]
            self.Lhat[m] = rides[n,m-1]

def saveGuest(fn, guests):
    jsdata  = []
    for g in sorted(guests.keys()):
        jsdata.append(guests[g].var_dump())
    fp  = open(fn, "w")
    fp.write(json.dumps(jsdata, indent=2))
    fp.close()

# def set_park(N, ddir,dependency={}, mode="linear", lam=False):

#     # とりあえず固定値
#     beta = list(range(1,10+1,1))
#     mu = 7.5 # 平均30分(1800秒)で裾が6時間ぐらいまで広がるはず
#     # mu = 8.2 # 移動時間も含めて平均60分ぐらい
#     sigma = 0.7

#     guest_feature = ( beta, mu, sigma )
#     guests      = {}
#     for i in range(1, N+1):
#         if mode == "linear":
#             g   = Guest_linear(i, attractions, 4)
#             g.set_alpha(guest_feature)
#         else:
#             g   = Guest(i, attractions, 4,N,lam=lam)
#         guests[i]   = g
#     saveGuest("%s/guest.json"%ddir, guests)
#     shutil.copy2("distance.csv", ddir)

def str2list_float(a):
    a = a.split(",")
    a = map(float, a)
    a = list(a)
    return a

def zero_truncated_poisson(lam):
    # ゼロ切断ポアソン分布
    while True:
        x = np.random.poisson(lam)
        if x > 0:
            break
    return x

def set_park_beta(N, basedir, ddir, config):
    # config = configparser.ConfigParser()
    # config.read(configfn)
    max_iteration = config.getint('SIMULATION', 'max_iteration')
    guests      = {}
    for i in range(1, N+1):
        # num_wish = 4
        num_wish = zero_truncated_poisson(3)
        # while True:
        #     num_wish = np.random.poisson(3)
        #     if num_wish > 0:
        #         break
        # if mode == "linear":
        # g   = Guest_beta(i, attractions, num_wish,N)
        g   = Guest_beta(i, num_wish,N)
        #     # g   = Guest_linear(i, attractions, num_wish,N,dependency=dependency)
        g.set_born(max_iteration)
        g.set_alpha(config)
        g.set_Lhat(config)
        # else:
        #     g   = Guest(i, attractions, num_wish, N, lam=lam)
        guests[i]   = g
    saveGuest("%s/guest.json"%ddir, guests)
    shutil.copy2("%s/distance.csv"%basedir, ddir)
    shutil.copy2("%s/attraction.csv"%basedir, ddir)
    shutil.copy2("config.ini", ddir)

def set_park_copy(N, basedir, ddir, config):
    # config = configparser.ConfigParser()
    # config.read(configfn)
    max_iteration = config.getint('SIMULATION', 'max_iteration')
    alphas = np.loadtxt("%s/alphas.csv"%basedir, delimiter=",")
    rides = np.loadtxt("%s/rides.csv"%basedir, delimiter=",")
    guests      = {}
    for i in range(1, N+1):
        # num_wish = 4
        while True:
            num_wish = np.random.poisson(3)
            if num_wish > 0:
                break
        g   = Guest_copy(i, num_wish,N)
        g.set_born(max_iteration)
        g.set_alpha_Lhat(alphas, rides)
        guests[i]   = g
    saveGuest("%s/guest.json"%ddir, guests)
    shutil.copy2("%s/distance.csv"%basedir, ddir)
    shutil.copy2("%s/attraction.csv"%basedir, ddir)
    shutil.copy2("config.ini", ddir)

def sort_park(ddir, outdir, dependency={}):
    shutil.copy2("distance.csv", outdir)
    shutil.copy2("%s/attraction.json"%ddir, outdir)

    guests      = {}
    guestfn = "%s/guest.json"%ddir
    fp  = open(guestfn)
    guests  = json.load(fp)
    fp.close()
#    N = len(guests)
#    print(guests)
#    for i in range(1, N+1):
    for g in guests:
#        g   = Guest(i, attractions, 4,N,dependency=dependency)
#        print(g["wish"])
        g["wish"] = topological_sort(g["wish"],dependency)
    fp = open("%s/guest.json"%outdir, 'w')
    fp.write(json.dumps(guests, indent=2))
    fp.close()

def copy_park(ddir, outdir):
    # shutil.copytree(ddir, outdir) # 既にフォルダがあるとエラー
    # shutil.copy2("%s/max_iteration.csv"%ddir, outdir)
    shutil.copy2("%s/distance.csv"%ddir, outdir)
    shutil.copy2("%s/attraction.csv"%ddir, outdir)
    shutil.copy2("%s/guest.json"%ddir, outdir)
    # shutil.copy2("%s/config.ini"%ddir, outdir)

def delay_park(ddir, outdir):
    shutil.copy2("distance.csv", outdir)
    shutil.copy2("%s/attraction.json"%ddir, outdir)

    guests      = {}
    guestfn = "%s/guest.json"%ddir
    fp  = open(guestfn)
    guests  = json.load(fp)
    fp.close()
#    N = len(guests)
#    print(guests)
#    for i in range(1, N+1):
    for g in guests:
#        g   = Guest(i, attractions, 4,N,dependency=dependency)
#        print(g["wish"])
        # if g["born"] > 3000 and g["born"] < 6000:
        #     g["born"] = 6000
        if g["born"] > 6000:
            g["born"] += 3000
    fp = open("%s/guest.json"%outdir, 'w')
    fp.write(json.dumps(guests, indent=2))
    fp.close()

def calc_throughput():
    # fp = open("results/20190621_145429/N3000it0modenormallam0.3/attraction.json")
    fp = open("/home/shimizu/project/2019/sim_park/results/20190703_151423/N4000it0beta0modenormal/attraction.json")
    attractions = json.load(fp)
    cum_throughpht = 0
    print(attractions)
    for att in attractions:
        print(att[u'throughput'])
        cum_throughpht += att[u'throughput']
        # print(att[u"throughpht"])
    print(cum_throughpht)
    # 0.501115326553 people per step


# if __name__ == '__main__':
#     calc_throughput()
# #    N = 1000
# #    set_park(N)
#     #

#     # sort_park("results/20190619_134541/N3000it10modesort", "tmp")
#     sys.exit()


#     attraction = np.loadtxt("attraction.csv", delimiter=",")
#     print(attraction)
#     M = attraction.shape[0]
#     distance_matrix = np.loadtxt("distance.csv",delimiter=",")
#     print(distance_matrix)

#     attractions = {}
#     for i in range(1, M+1):
#         a   = Attraction(i, st=attraction[i-1,0], capa=attraction[i-1,1])
#         attractions[i]=a

#     saveAttraction("attraction.json",attractions)

# #    N = 2000
#     Ns = range(1000,6000,1000)
#     for N in Ns:
#         if not os.path.exists("settings/%d"%N):
#             os.makedirs("settings/%d"%N)
#         guests      = {}
#         for i in range(1, N+1):
#             g   = Guest(i, attractions, 4,N)
#             guests[i]   = g
#         saveGuest("settings/%d/guest.json"%N, guests)
# #        shutil.copy2("")
