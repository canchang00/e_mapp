import os
import argparse

import time
import sys

import numpy as np
import random
from collections import namedtuple
from envs.overcooked_environment import OvercookedEnvironment, ProgramEnvironment
from envs.multi_env_wrapper import SubprocVecEnv

import torch
import cv2

from common.utils import *
import yaml

from algorithm.rl.ppo import PPO


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")
    parser.add_argument("--random_action_prob", type=float, default=0, help="Possibility that your action will be replaced by a random action")

    # Reward 
    parser.add_argument("--dense_reward", action="store_true",default=False)
    
    parser.add_argument("--exp_name", type=str, default="default")
    
    parser.add_argument("--exit_program_penalty", action="store_true", default=False)
    
    parser.add_argument("--no_program", action="store_true", default=False)
    
    parser.add_argument("--use_ppg", action="store_true", default=False)
    parser.add_argument("--force_follow_program", action="store_true", default=False)
    
    parser.add_argument("--cfg_file", type=str, required=True)
    parser.add_argument("--test_only", action="store_true", default=False)
    
    return parser.parse_args()

def setup_seed(seed):
    if not seed==-1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
if __name__=="__main__":
    arglist = parse_arguments()
    setup_seed(arglist.seed)
    exp_name=arglist.exp_name
    record_time=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 
    project_name="final"
    log_dir=os.path.join("logs",project_name,exp_name,record_time)
    os.makedirs(log_dir,exist_ok=True)
    with open(os.path.join(log_dir,"arguments.txt"),"w") as f:
        f.write(str(arglist))
    
    with open(arglist.cfg_file) as f:
        try:
            cfg = AttrDict(yaml.load(f, Loader=yaml.SafeLoader))
        except yaml.YAMLError as e:
            print(e)
        env=ProgramEnvironment(cfg=cfg,arglist=arglist,subtask_supervised=True,exit_program_penalty=arglist.exit_program_penalty,use_program=(not arglist.no_program),num_agents=cfg.num_agents)
        vec_env=SubprocVecEnv(env,cfg.num_envs)
        eval_env=SubprocVecEnv(env,1)
    
    

    model=PPO(env=vec_env,eval_env=eval_env,log_dir=log_dir,**vars(arglist))
    
    # evaluating the model on the test scenes, save testing audio as "demo.avi"
    if arglist.test_only:
        model.evaluate_mamt(list(range(1)))
        exit(0)
    
    model.learn(total_timesteps=int(3e6))
    env.close()
