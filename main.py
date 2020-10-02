import argparse
import gym
import numpy as np
# from envs import Curriculum
from envs import Environment
# from envs import Curriculum
from myenv import env

parser = argparse.ArgumentParser(description="A2C")
parser.add_argument('--seed', type=int, default=1, 
        help='random seed')
# parser.add_argument('--num-episodes', type=int, default=100, 
#         help='The number of episodes')
# parser.add_argument('--num-processes', type=int, default=16, 
#         help='The number of processes')
# parser.add_argument('--num-advanced-step', type=int, default=34, 
#         help='The number of advanced step')
parser.add_argument('--step', type=int, default=0, 
        help='step')
parser.add_argument('--save-interval', type=int, default=5, 
        help='save interval')
# parser.add_argument('--loss-coef', type=float, default=0.5, 
#         help='Loss coeffient')
# parser.add_argument('--entropy-coef', type=float, default=0.01, 
#         help='Entropy coeffient')
# parser.add_argument('--max-grad-norm', type=float, default=0.5, 
#         help='Max grad norm')
# parser.add_argument('--gamma', type=float, default=0.99, 
#         help='discount rate')
parser.add_argument('--env-name', type=str, default='simenv-v1', 
        help='Environments')
parser.add_argument('--inputfn', type=str, default="model", 
        help='model file path')
parser.add_argument('--configfn', type=str, default="config.ini", 
        help='config file path')
# parser.add_argument('--outputfn', type=str, default="model", 
#         help='output file name')
# parser.add_argument('--resdir', type=str, default="logs", 
#         help='file path of results')
parser.add_argument('--checkpoint', action='store_true',
        help='is load model')
parser.add_argument('--save', action='store_true',
        help='is save model')
parser.add_argument('--test', action='store_true',
        help='test mode')
parser.add_argument('--cuda', action='store_true',
        help='when you use cuda, you enable this option.')

if __name__ == '__main__':
    args = parser.parse_args()

    env = Environment(args, flg_test=True)
    S_open = env.test()
    print("誘導なし", S_open)
    env = Environment(args, flg_test=False, S_open=S_open)
    model = env.train()

    env = Environment(args, flg_test=True)
    print(env.test(model))


    """
    env = gym.make('simenv-v0')
    env.reset()

    done    = False
    step    = 0
    rewards = 0
    actions = [0, 4, 5, 4, 3, 4, 5, 4, 5, 5, 5, 1, 3, 2, 2, 3, 2, 6, 3, 5, 6, 5, 4, 2, 5, 3, 1, 6, 1, 4, 3, 5, 3, 0]

    while not done:
        # action = np.random.choice([0, 1, 2, 3, 4, 5, 6])
        action = actions[step]
        observation, reward, done, _ = env.step(action)
        print("OBSERV", action, step, env.cur_time, np.sum(observation), reward, done)
        rewards += reward
        step    += 1
    print(rewards)
    """
