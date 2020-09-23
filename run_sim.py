#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 20:22:55 2019

@author: shimizu
"""
import itertools
import set_park
# import sim_park
from sim_park import SimPark
from sim_park import sim_park
#import estimate_set
from datetime import datetime
import os, sys, shutil
import numpy as np
#from copy_figure import copy_figure
import random
import copy
#from isCyclic import Graph
import pickle
import glob
from joblib import Parallel, delayed
from collections import defaultdict
from operator import attrgetter
import configparser
from copy import deepcopy
# import optuna
# from optuna.samplers import TPESampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

datetime_now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
seed = int( datetime_now[-2:] )

# target = "ST" # stay time
target = "GS" # guest surplus

# flg_parallel = False
flg_parallel = True

class Graph(): 
    # https://www.geeksforgeeks.org/detect-cycle-in-a-graph/
    def __init__(self,vertices): 
        self.graph = defaultdict(list) 
        self.V = vertices
  
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
  
    def isCyclicUtil(self, v, visited, recStack):   
        # Mark current node as visited and  
        # adds to recursion stack 
        visited[v] = True
        recStack[v] = True
  
        # Recur for all neighbours 
        # if any neighbour is visited and in  
        # recStack then graph is cyclic 
        for neighbour in self.graph[v]: 
            if visited[neighbour] == False: 
                if self.isCyclicUtil(neighbour, visited, recStack) == True: 
                    return True
            elif recStack[neighbour] == True: 
                return True
  
        # The node needs to be poped from  
        # recursion stack before function ends 
        recStack[v] = False
        return False
  
    # Returns true if graph is cyclic else false 
    def isCyclic(self): 
        visited = [False] * (self.V +1)
        recStack = [False] * (self.V +1)
        for node in range(1,self.V+1): 
            if visited[node] == False: 
                if self.isCyclicUtil(node,visited,recStack) == True: 
                    return True
        return False

    def matrixD(self):
        ret = np.zeros((self.V +1,self.V +1))
        # 1行目が，アトラクション1を体験した後に体験できる選択肢
        for i in range(1,self.V+1,1):
            for j in range(1,self.V+1,1):
                if j in self.graph[i]: # restriction
                    ret[i,j]=0 # not available
                else: # no restriction
                    ret[i,j]=1 # available
        return ret

    def matrixD4paper(self):
        ret = np.zeros((self.V +1,self.V +1))
        # 1行目が，アトラクション1を体験した後に体験できる選択肢
        for i in range(1,self.V+1,1):
            for j in range(1,self.V+1,1):
                if j in self.graph[i]: # restriction
                    ret[j,i]=1 # 論文のijとコード中のijが入れ替わっているので要注意
                else: # no restriction
                    ret[j,i]=0 # 論文では体裁を重視，コードは歴史的理由でij決めたため
        return ret[1:-1,1:-1] # グラフの大きすぎる部分を省略して表示

    def readMatrix(self, ):
        ret = np.zeros((self.V +1,self.V +1))
        # 1行目が，アトラクション1を体験した後に体験できる選択肢
        for i in range(1,self.V+1,1):
            tmp = []
            for j in range(1,self.V+1,1):
                if j in self.graph[i]: # restriction
                    ret[i,j]=0 # not available
                else: # no restriction
                    ret[i,j]=1 # available



def mk_ordered_graph(V,order):
    g = Graph(V+1)
    for i in range(1,V+1,1):
        for j in range(i+1,V+1,1):
            if order.index(i) > order.index(j):
                g.addEdge(j,i)
    # print(g.graph)
    return g

# def convert2graph(individual,V=10):
def convert2graph(individual,V=5):
    # print(individual)
    # g = Graph(V+1)
    g = Graph(V)
    for i in range(1,V+1,1):
        tmp = individual[(i-1)*V:i*V]
        for j in range(1,V+1,1):
            if tmp[j-1] == 1:
                g.addEdge(i,j)
    # print(g.graph)
    return g

def convert2ind(g,V=10):
    ret = []
    for i in range(1,V+1,1):
        tmp = [0] * V
        for j in range(1,V+1,1):
            if j in g.graph[i] :
                tmp[j-1] = 1
        ret.extend(tmp)
    # print(ret)
    ret = Individual(ret)
    # print(convert2graph(ret).graph)
    return ret

# from distutils.dir_util import copy_tree

def run_parallel(ddirs, Ds, config):
    # ret = []
    # results = Parallel(n_jobs=-1)([delayed(sim_park.sim_park)(ddir, ddir, "config.ini", mode="linear", dependency=D) for ddir, D in zip(ddirs,Ds)])
    # results = Parallel(n_jobs=-1)([delayed(sim_park.sim_park)(ddir, ddir, config, dependency=D) for ddir, D in zip(ddirs,Ds)])
    results = Parallel(n_jobs=-1)([delayed(sim_park)(ddir, ddir, config, dependency=D) for ddir, D in zip(ddirs,Ds)])
    # for i, result in enumerate(results):
    #     # total_exp, mean_exp, queue, mean_wait, mean_Inpark, nTime, mean_util = result
    #     ret.append(result)
        # ret.append(mean_util - queue)
    return results
    # return ret

def run_single(ddirs, Ds, config):
    results = []
    # ret = []
    for ddir, D in zip(ddirs,Ds):
        # result = sim_park.sim_park(ddir, ddir, config, dependency=D)
        result = sim_park(ddir, ddir, config, dependency=D)
        results.append(result)
        # results = Parallel(n_jobs=-1)([delayed(sim_park.sim_park)(ddir, ddir, "config.ini", mode="linear", dependency=D) for ddir, D in zip(ddirs,Ds)])
        # results = Parallel(n_jobs=-1)([delayed(sim_park.sim_park)(ddir, ddir, "config.ini", mode="linear", dependency=D) for ddir, D in zip(ddirs,Ds)])
    # for i, result in enumerate(results):
    #     # total_exp, mean_exp, queue, mean_wait, mean_Inpark, nTime, mean_util = result
    #     ret.append(result)
        # ret.append(mean_util - queue)
    return results

# class Individual(np.ndarray):
class Individual():
    """Container of a individual."""
    # def __new__(cls, a):
    #     return np.asarray(a).view(cls)
    def __init__(self, a):
        self.fitness = None
        self.gene = np.asarray(a).copy()

    def set_f(self, f):
        self.fitness = copy.deepcopy(f)

    def mutFlipBit(self, indpb):
        """Mutation function."""
        tmp = self.gene.copy()
        for i in range(len(self.gene)):
            if random.random() < indpb:
                tmp[i] = type(self.gene[i])(not self.gene[i])
        self.gene = tmp.copy()
        self.fitness = None # 評価は初期化

    def set_objective(self, ddirs, GAdir, config): # to set sim env
        self.Obj = Objective(ddirs, config)
        # Obj.set_resdir(GAdir)
        self.Obj.set_sims(GAdir,config)

    def set_fitness(self, dict_fitness): # to evaluate
        if tuple(self.gene) in dict_fitness:
            fit = dict_fitness[tuple(self.gene)]
            # self.fitness = dict_fitness[tuple(self.gene)]
        else:
            # self.fitness = self.Obj.objective_gene(self.gene)
            fit = self.Obj.objective_gene(self.gene)
            # dict_fitness[tuple(self.gene)] =  copy.deepcopy( self.fitness )
        #     self.set_f( dict_fitness[tuple(self.gene)])
        #     return dict_fitness
        # score = self.Obj.objective_gene(self.gene)
        # self.set_f( score )
        # dict_fitness[tuple(self.gene)] =  copy.deepcopy( score )
        return fit



def selTournament(pop, n_ind, tournsize):
    """Selection function."""
    chosen = []    
    for i in range(n_ind):
        aspirants = [random.choice(pop) for j in range(tournsize)]
        chosen.append(max(aspirants, key=attrgetter("fitness")))
    return chosen

def cxTwoPointCopy(ind1, ind2):
    """Crossover function."""
    size = len(ind1.gene)
    tmp1 = deepcopy( ind1 )
    tmp2 = deepcopy( ind2 )
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size-1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        tmp_point = deepcopy( cxpoint1 )
        cxpoint1 = deepcopy( cxpoint2 )
        cxpoint2 = deepcopy( tmp_point )

    tmp1.gene[cxpoint1:cxpoint2] = ind2.gene[cxpoint1:cxpoint2]
    tmp2.gene[cxpoint1:cxpoint2] = ind2.gene[cxpoint1:cxpoint2]
    return tmp1, tmp2

def write_dir_list(fn, ddirs):
    with open(fn, "w") as f:
        for ddir in ddirs:
            f.write("%s\n"%ddir)

def run_sim_GA(config, basedir):
    for N in range(1000,10000,1000):
    # for N in [1000]:
        # ddirs = sorted( glob.glob("results/normal10/N%sit*"%N) )
        ddirs = sorted( glob.glob("data/beta/N%sit*"%N) )
        GAdir = "%s/N%s"%(basedir, N)
        os.makedirs(GAdir, exist_ok=True)
        write_dir_list("%s/dir_list.txt"%GAdir, ddirs)
        print(ddirs)
        sim_GA(ddirs, GAdir, config)

def run_sim_GA_alldirs(config, basedir):
    # for N in range(1000,10000,1000):
    # for N in [1000]:
    # ddirs = sorted( glob.glob("results/normal10/N%sit*"%N) )
    ddirs = sorted( glob.glob("data/beta/N?000it*") )
    GAdir = basedir
    os.makedirs(GAdir, exist_ok=True)
    write_dir_list("%s/dir_list.txt"%GAdir, ddirs)
    print(ddirs)
    sim_GA(ddirs, GAdir, config)

def sim_GA(ddirs,GAdir, config):
    # ddirs : 来園者データを入っているフォルダのリスト
    # f : file pointer
    f = open("%s/GA_log.txt"%GAdir, "w")
    n_ind    = int(config['GA']['n_ind']) # 72   # The number of individuals in a population.
    CXPB     = float(config['GA']['CXPB']) # 0.1   # The probability of crossover.
    MUTPB    = float(config['GA']['MUTPB']) # 0.2   # The probability of individdual mutation.
    MUTINDPB = float(config['GA']['MUTINDPB']) # 0.1  # The probability of gene mutation.
    NGEN     = int(config['GA']['NGEN']) # 100    # The number of generation loop.

    num_att = int(config['SIMULATION']['num_attraction'])
    n_gene = num_att ** 2

    dict_fitness = {}
    random.seed(seed)
    # random.seed(64)
    # --- Step1 : Create initial generation.
    pop = []
    for i in range(n_ind):
        ind = Individual([np.random.binomial(n=1, p=0.1) for i in range(n_gene)])
        # ind = list( np.random.choice([0,1], 10, p=[0.8,0.2]) )
        # ind = list( np.random.randint(0,2,n_gene) ) # 重みなし
        logdir = "%s/tmp%02d"%(GAdir, i) # n_ind < 100を想定
        ind.set_objective(ddirs, logdir, config)
        pop.append(ind)

    results = Parallel(n_jobs=-1)([delayed(p.set_fitness)(dict_fitness) for p in pop])
    for p, result in zip(pop, results):
        fit = result
        dict_fitness[tuple(p.gene)] = fit
        p.fitness = fit
    # for i,p in enumerate(pop):
    #     print(i, p.fitness)
    #     dict_fitness = p.set_fitness(dict_fitness)
    best_ind = max(pop, key=attrgetter("fitness"))
    ever_best_ind = copy.deepcopy(best_ind)
    ever_best_ind.set_f(best_ind.fitness)
    # --- Generation loop.
    print("Generation loop start.")
    f.write("Generation loop start.\n")
    # print("Generation: 0. Best fitness: " + str(best_ind.fitness))
    # print("Generation: 0. Best fitness: " + str(1.0 / best_ind.fitness))
    print("Generation: 0. Best fitness: " + str( best_ind.fitness))
    f.write("Generation: 0. Best fitness: " + str( best_ind.fitness) + "\n")
    print(convert2graph(best_ind.gene).graph)
    for g in range(NGEN):
        
        # --- Step2 : Selection.
        offspring = selTournament(pop, n_ind, tournsize=3)
        
        # --- Step3 : Crossover.
        crossover = []
        cnt_crossover = 0
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                tmp_child1, tmp_child2 = cxTwoPointCopy(child1, child2)
                child1 = tmp_child1
                child2 = tmp_child2
                child1.fitness = None
                child2.fitness = None
                cnt_crossover += 1
            crossover.append(child1)
            crossover.append(child2)
        print("crossover: %d"%cnt_crossover)
        # print(len(crossover))
        offspring = crossover[:]
        # print(offspring)
        
        # --- Step4 : Mutation.
        mutant = []
        cnt_mutant = 0
        for mut in offspring:
            if random.random() < MUTPB:
                mut.mutFlipBit(indpb=MUTINDPB)
                # mut.fitness = None
                cnt_mutant += 1
            mutant.append(mut)
        print("mutant: %d"%cnt_mutant)

        offspring = mutant[:]
        
        # --- Update next population.
        pop = offspring[:]
        results = Parallel(n_jobs=-1)([delayed(p.set_fitness)(dict_fitness) for p in pop])
        for p, result in zip(pop, results):
            fit = result
            dict_fitness[tuple(p.gene)] = fit
            p.fitness = fit
        # for p in pop:
        #     # pop, dict_fitness = set_fitness(pop, dict_fitness, ddirs, config, GAdir)
        #     dict_fitness = p.set_fitness(dict_fitness)
        
        # --- Print best fitness in the population.
        best_ind = max(pop, key=attrgetter("fitness"))
        if best_ind.fitness > ever_best_ind.fitness:
            ever_best_ind = copy.deepcopy(best_ind)
            ever_best_ind.set_f(best_ind.fitness)
        # print("Generation: " + str(g+1) + ". Best fitness: " + str(best_ind.fitness))
        # print("Generation: " + str(g+1) + ". Best fitness: " + str(1.0 / best_ind.fitness))
        print("Generation: " + str(g+1) + ". Best fitness: " + str( ever_best_ind.fitness))
        f.write("Generation: " + str(g+1) + ". Best fitness: " + str( ever_best_ind.fitness) + "\n")
        print(convert2graph(best_ind.gene).graph)
        f.write(str(convert2graph(best_ind.gene).graph)+"\n")
    
    print("Generation loop ended. The best individual: "+ str( ever_best_ind.fitness) )
    f.write("Generation loop ended. The best individual:"+ str( ever_best_ind.fitness) + " \n")
    # print(best_ind)
    # f.write(str(best_ind))
    D = convert2graph(ever_best_ind.gene)
    print(D.graph)
    f.write(str(D.graph)+"\n")
    with open('%s/matrixD.pickle'%GAdir, mode='wb') as fp:
        pickle.dump(D, fp)
    # with open('%s/best_%s_ind.pickle'%(GAdir,target), 'wb') as fp:
    #     pickle.dump(ever_best_ind, fp)
    f.close()

def run_sim(config, ddirs, rdir=None, D=None): # normalとminWTとmaxTLで共通にする
    # ddirs: 来園者データを格納したフォルダのリスト
    # rdir: ログ出力フォルダ
    if D is None:
        D = Graph(5) # No dependency
    if rdir is None:
        rdir  = config['SIMULATION']['logdir']
    if(not os.path.exists(rdir)):
        os.mkdir(rdir)
    shutil.copy2(__file__, rdir)
    shutil.copy2("set_park.py", rdir)
    shutil.copy2("sim_park.py", rdir)
    f = open("%s/log.txt"%rdir, "w")
    Ds = [D] * len(ddirs) # 制約なし
    resdirs = []
    for ddir in ddirs:
        tmp_name = ddir.split("/")[-1]
        resdir = "%s/%s"%(rdir, tmp_name)
        if(not os.path.exists(resdir)):
            os.mkdir(resdir)
        set_park.copy_park(ddir,resdir)
        resdirs.append(resdir)

    print("%d simulations to run"%len(ddirs))
    if flg_parallel:
        results = run_parallel(resdirs, Ds, config)
    else:
        results = run_single(resdirs, Ds, config)
    out = "tmp_name, total_exp, mean_exp, queue, mean_wait, mean_Inpark, nTime, mean_util, mean_surplus"
    print(out)
    f.write("%s\n"%out)
    for resdir,result in zip(resdirs,results):
        tmp_name = resdir.split("/")[-1]
        out = tmp_name
        for res in result:
            out += " %.2f"%res
        print(out)
        f.write("%s\n"%out)
    f.close()

def run_setting(config, targetdir): # set_parkの部分を切り離す
    # targetdir: 生成したファイルを保存するフォルダ
    # ddir: targetdirの配下の来園者数ごとのフォルダ
    # basedir: attraction.csvとdistance.csvがあるフォルダ
    # 注意：set_park_betaとset_park_copyの使い分けが手作業

    Ns = range(1000,10000,1000)
    # Ns = range(10000,20000,1000)
    its = range(1)
    basedir  = config['SIMULATION']['ddir']
    ddirs = []
    for it in its:
        for n in Ns:
            # for b, beta in enumerate(betas):
            tmp_name ="N%sit%s"%(n,it)
            print(tmp_name)
            # ddir  = "%s/%s"%(basedir,tmp_name)
            ddir  = "%s/%s"%(targetdir,tmp_name)
            os.makedirs(ddir,exist_ok=True)
            set_park.set_park_beta(n,basedir,ddir,config)
            # set_park.set_park_copy(n,basedir,ddir,config)
            ddirs.append(ddir)
            print(it, n, ddir)
    return ddirs

def run_sim_perm(config, ddirs):
    # たしか巡回順指定の評価
    M = int( config['SIMULATION']['num_attraction'] )
    permDs = mk_perm(M)
    basedir  = "results/perm"
    for d, D in enumerate(permDs):
        rdir  = "%s/D%s"%(basedir,d)
        print(rdir)
        print_D(D, M)
        if(not os.path.exists(rdir)):
            os.makedirs(rdir)
        with open('%s/matrixD.pickle'%rdir, mode='wb') as f:
            pickle.dump(D, f)
        run_sim(config, ddirs, rdir=rdir, D=D)

# def run_sim_replay(config, ddirs):
#     # たしか巡回順指定の評価
#     M = int( config['SIMULATION']['num_attraction'] )
#     permDs = mk_perm(M)
#     basedir  = "results/perm"
#     for d, D in enumerate(permDs):
#         rdir  = "%s/D%s"%(basedir,d)
#         print(rdir)
#         print_D(D, M)
#         if(not os.path.exists(rdir)):
#             os.makedirs(rdir)
#         with open('%s/matrixD.pickle'%rdir, mode='wb') as f:
#             pickle.dump(D, f)
#         run_sim(config, ddirs, rdir=rdir, D=D)

def print_D(D,V):
    # 1行目が，アトラクション1を体験した後に体験できる選択肢
    for i in range(1,V+1,1):
        tmp = []
        for j in range(1,V+1,1):
            if j in D.graph[i]: # restriction
                tmp.append("0") # not available
            else: # no restriction
                tmp.append("1") # available
        print(" ".join(tmp))

def mk_perm_D(perm):
    # permはアトラクションIDのリスト
    M = len(perm)
    ret = Graph(M+1)
    for i in range(1,M+1,1):
        for j in range(1,M+1,1):
            # 後のアトラクションに乗ったら前のアトラクションには乗れない
            if perm.index(j) < perm.index(i):
                ret.addEdge(i,j) # iに乗ったらjに乗れないフラグ
    return ret

def mk_perm(M):
    # M = 5
    Ds = []
    perms = list(itertools.permutations(range(1,M+1)))
    for perm in perms: # permはアトラクションIDのリスト
        print(perm)
        tmp = mk_perm_D(perm)
        Ds.append(tmp)
    return Ds

def load_print_D(fn):
    with open(fn, mode='rb') as f:
        D = pickle.load(f)
    print_D(D, 5)

def test_D():
    load_print_D("results/perm/D0/matrixD.pickle")
    # mk_perm(5)
    sys.exit()

class Objective():
    def __init__(self, ddirs, config):
        self.ddirs = ddirs
        self.config = config
        # self.i = 0
        # self.num_parallel = 100
        M = int( config['SIMULATION']['num_attraction'] )
        self.d = M * M # 25

    # def set_resdir(self, resdir):
    #     # while True:
    #     #     i = np.random.randint(self.num_parallel)
    #     #     resdir = "results/tmp%02d"%i
    #     if not os.path.exists(resdir):
    #         # break
    #         os.makedirs(resdir)
    #     self.resdirs = []
    #     for i, ddir in enumerate( self.ddirs ):
    #         rdir = "%s/tmp%s"%(resdir, i)
    #         os.makedirs(rdir, exist_ok=True)
    #         # set_park.copy_park(ddir, rdir)
    #         self.resdirs.append(rdir)

    def set_sims(self, resdir, config):
        self.sims = []
        if not os.path.exists(resdir):
            # break
            os.makedirs(resdir)
        self.resdirs = []
        for i, ddir in enumerate( self.ddirs ):
            rdir = "%s/tmp%s"%(resdir, i)
            os.makedirs(rdir, exist_ok=True)
            # set_park.copy_park(ddir, rdir)
            self.resdirs.append(rdir)
            sim = SimPark(ddir, rdir, config)
            self.sims.append(sim)

    def objective(self, trial):
        x = np.array( [int( trial.suggest_discrete_uniform('x_'+str(i), 0, 1, 1)) for i in range(self.d) ] )
        D = convert2graph(x)
        tmp_scores = []
        # for ddir, resdir in zip( self.ddirs, self.resdirs) :
        #     total_exp, mean_exp, queue, mean_wait, mean_Inpark, nTime, mean_util, mean_surplus = sim_park.sim_park(ddir, resdir, self.config, dependency=D)
        #     tmp_scores.append(mean_surplus)
        for sim in self.sims:
            print(sim.ddir)
            sim.simulate(D)
            tmp_scores.append(sim.evaluate())
        # if target == "ST": # stay time
        #     score = - mean_Inpark
        # else: # for comparison: GS for guest surplus
        # score = mean_util - queue
        score = np.mean(tmp_scores)
        print(tmp_scores, score)
        # shutil.rmtree(self.resdir)
        return -score

    def objective_gene(self, gene):
        D = convert2graph(gene)
        tmp_scores = []
        for sim in self.sims:
            # print(sim.ddir)
            sim.simulate(D)
            tmp_scores.append(sim.evaluate())
        score = np.mean(tmp_scores)
        print(tmp_scores, score)
        return score


def run_sim_optuna(config, basedir):
    for N in range(1000,5000,1000):
    # for N in range(1000,10000,1000):
        # for N in [2000]:
        ddirs = sorted( glob.glob("results/normal10/N%sit*"%N) )
        GAdir = "%s/N%s"%(basedir, N)
        os.makedirs(GAdir, exist_ok=True)
        write_dir_list("%s/dir_list.txt"%GAdir, ddirs)
        print(ddirs)
        sim_optuna(ddirs, GAdir, config)

def sim_optuna(ddirs,GAdir, config):
    # ddirs : 来園者データを入っているフォルダのリスト
    # f : file pointer
    NGEN     = int(config['optuna']['NGEN']) # 100    # The number of generation loop.

    Obj = Objective(ddirs, config)
    # Obj.set_resdir(GAdir)
    Obj.set_sims(GAdir,config)
    study = optuna.create_study(sampler=TPESampler())
    # study.optimize(Obj.objective, n_trials=NGEN, n_jobs=-1)
    study.optimize(Obj.objective, n_trials=NGEN, n_jobs=1)
    print('params:', study.best_params)
    df = study.trials_dataframe()
    print(df['value'])

    plt.plot(df['value'])
    plt.savefig('%s/test_optuna.png'%GAdir)

    print(study.best_value)

    # with open('%s/best_params.pickle'%(logdir), 'wb') as fp:

    M = int( config['SIMULATION']['num_attraction'] )
    d = M * M # 25
    x = np.array( [int( study.best_params['x_'+str(i)]) for i in range(d) ] )
    D = convert2graph(x)
    matrixD = D.matrixD()
    # np.savetxt("%s/best_params.txt"%(GAdir),x,fmt="%d")
    np.savetxt("%s/best_params.txt"%(GAdir),matrixD,fmt="%d")
    np.savetxt("%s/best_value.txt"%(GAdir),np.array([study.best_value]),fmt="%f")
    with open('%s/best_%s_ind.pickle'%(GAdir,target), 'wb') as fp:
        pickle.dump(D, fp)

def test_class_function():
    configfn = "config_normal.ini"
    config = configparser.ConfigParser()
    config.read(configfn)
    dependency = Graph(5)

    # class
    sim = SimPark("data/N1000it0", "results/tmp_class", config)
    sim.simulate(dependency)
    sim.saveLog()
    print(sim.evaluate())

    # function
    result = sim_park("data/N1000it0", "results/tmp_function", config, dependency)
    print(result[-1])

def run_sim_replay_perm(exe_flg=False):
    configfn = "config_normal.ini"
    config = configparser.ConfigParser()
    config.read(configfn)
    D = {}
    lines = open("results/perm/summary.txt").readlines()
    for line in lines:
        line = line.strip().split(" ")
        N = int( line[0] )
        
        # Dfn = "results/20200509perm/%s/matrixD.pickle"%(line[1])
        Dfn = "%s/matrixD.pickle"%(line[1])
        # D[N] = Graph(5)
        D[N] = pickle.load(open(Dfn, "rb"))
        # D[N] = pickle.load(Dfn)
        # print(D[N])
        print(N)
        # print(D[N].V)
        # print(D[N].graph)
        print(D[N].matrixD())
    if not exe_flg:
        sys.exit()

    for N in range(1000, 10000, 1000):
        # ddirs = glob.glob("results/normal10/N%sit?"%N) # 共有計算機でGAの実験をやった，これをベースにする
        ddirs = glob.glob("data/beta/N%sit?"%N) # 制約変更の際に，来園者再作成，これをベースにする
        rdir = "results/replay_perm/N%s"%N
        os.makedirs(rdir, exist_ok=True)
        run_sim(config, ddirs, rdir=rdir, D=D[N])

def run_sim_replay_GA(exe_flg=False):
    configfn = "config_normal.ini"
    config = configparser.ConfigParser()
    config.read(configfn)
    D = {}
    for N in range(1000, 10000, 1000):
        # Dfn = "c_park_ln/20200511GA/N%s/best_GS_ind.pickle"%(N)
        Dfn = "results/GA/N%s/matrixD.pickle"%(N) # numpy行列で保存したつもりがgraphだった
        D[N] = pickle.load(open(Dfn, "rb"))
        # print(D[N])
        print(N)
        print(D[N].matrixD())
    if not exe_flg:
        sys.exit()

    for N in range(1000, 10000, 1000):
        # ddirs = glob.glob("results/normal10/N%sit?"%N) # 共有計算機でGAの実験をやった，これをベースにする
        ddirs = glob.glob("data/beta/N%sit?"%N) # 制約変更の際に，来園者再作成，これをベースにする
        rdir = "results/replay_GA/N%s"%N
        os.makedirs(rdir, exist_ok=True)
        run_sim(config, ddirs, rdir=rdir, D=D[N])

if __name__ == '__main__':
    # run_sim_replay_optuna()
    # run_sim_replay_perm()
    # run_sim_replay_GA()
    # sys.exit()

    # test_class_function()
    # sys.exit()

    configfn = "config.ini"
    # configfn = "config_normal.ini"
    # configfn = "config_minWT.ini"
    # configfn = "config_maxTL.ini"
    # configfn = "config_maxGS.ini"
    # configfn = "config_perm.ini" # normalをコピーして暫定作成20200721
    config = configparser.ConfigParser()
    config.read(configfn)
    # datehead = "normal"

    # 来園者データ生成
    # copyとbetaをrun_setting内のコメントアウトで選択しているので要注意
    targetdir = "data/beta"
    # targetdir = "data/copy"
    ddirs = run_setting(config, targetdir) # 新規に来園者を生成する場合はこれ

    # ddirs = glob.glob("data/copy/N1?000it0") # アンケートの複製
    # run_sim(config, ddirs)

    # ddirs = glob.glob("data/beta/N1?000it0") # 許容限界モデル
    # ddirs = glob.glob("data/beta/N?000it?") # 許容限界モデル
    # ddirs = glob.glob("data/beta/N?000it0") # 許容限界モデル
    # run_sim(config, ddirs)

    # ddirs = glob.glob("data/beta/N*it0") # 本当はこっちを使いたいけど
    # ddirs = glob.glob("results/normal/N?000it?") # GAの実験をやった，これはit0しかない
    # ddirs = glob.glob("results/normal10/N?000it?") # 共有計算機でGAの実験をやった，これをベースにする
    # ddirs = glob.glob("results/normal10/N1000it?") # 共有計算機でGAの実験をやった，これをベースにする

    # 誘導なし，minWTなどの実行
    run_sim(config, ddirs)

    # 順列誘導の実行
    # run_sim_perm(config, ddirs)

    # datehead = run_sim_normal(config)

    # GA探索する
    # run_sim_GA(config, "results/GA")
    # run_sim_GA_alldirs(config, "results/GA_alldirs")

    # optunaで探索する
    # run_sim_optuna(config, "results/optuna")
