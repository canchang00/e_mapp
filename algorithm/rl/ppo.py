import warnings
from typing import Any, Dict, Optional, Type, Union, NamedTuple, List
from cv2 import threshold

import numpy as np
import torch
from gym import spaces
from torch._C import device
from torch.nn import functional as F

from algorithm.rl.ppo_utils import RolloutBuffer, get_schedule_fn

import torch.nn as nn
import time

from collections import deque
from copy import deepcopy

from torch.distributions import Bernoulli, Categorical, Normal

from envs.multi_env_wrapper import SubprocVecEnv
from  envs.overcooked_environment import subtask

import random

import cv2
        
import os

import copy
                
from torch._six import inf
        
        
from functools import partial

import yaml
from common.utils import *
from gym import Env
import math

from munkres import Munkres
mnkrs=Munkres()

global rnd_buffer
rnd_buffer=np.random.rand(100,1)

global is_warmup

use_others_obs=True
    
class CategoricalMasked(Categorical):

    def __init__(self, logits, epsilon_greedy=0,mask=None):
        self.mask = mask
        if mask is not None and len(mask.shape)<2:
            self.mask=torch.unsqueeze(mask,0)
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            probs=F.softmax(logits, dim=1)
            masked_uniform=torch.ones_like(logits)*self.mask/torch.sum(self.mask,dim=1,keepdim=True)
            probs=probs*(1-epsilon_greedy)+masked_uniform*epsilon_greedy
            super(CategoricalMasked, self).__init__(probs=probs)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        p_log_p = self.logits*self.probs
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -torch.sum(p_log_p, dim=1)

    
class Policy_Overcooked(nn.Module):
    def __init__(self,cfg):
        super(Policy_Overcooked, self).__init__()
        self.cfg=cfg
        self.training_stage=cfg.training_stage
        self.num_agents=cfg.num_agents
        self.lr=cfg.lr
        self.len_task_repr=20
        self.num_objects=20
        
        self.cnn_encoder1=nn.Sequential(nn.LayerNorm([1,8,8]),
                                        nn.Conv2d(1,32,kernel_size=5,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                        nn.Flatten())  
        
        self.cnn_encoder_before_attention=nn.Sequential(nn.LayerNorm([self.num_objects,8,8]),nn.Conv2d(self.num_objects,64,kernel_size=3,stride=1,padding=1))
        self.cnn_encoder11=nn.Sequential(nn.Conv2d(64,64,kernel_size=5,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                        nn.Flatten()) 
        self.idx_embedding=nn.Embedding(num_embeddings=1,embedding_dim=32)
        self.terrain_encoder_v=nn.Linear(2304,128)
        self.terrain_encoder_p=nn.Linear(2304,128)
        self.obj_encoder_v=nn.Linear(2304,128)
        self.obj_encoder_p=nn.Linear(2304,128)
        
        self.map_encoding_dim=256
        self.state_encoding_dim=self.map_encoding_dim
        self.state_encoding_dim+=32
        self.agent_encoder_v=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,32))
        self.agent_encoder_p=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,32))
        self.recipe_encoder_v=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,32))
        self.recipe_encoder_p=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,32))

        self.modulation_dim=128+128+32+32
            
        self.subtask_encoder_v=nn.Sequential(nn.Linear(self.len_task_repr,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,self.modulation_dim*4))
        self.subtask_encoder_p=nn.Sequential(nn.Linear(self.len_task_repr,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,self.modulation_dim*4))
        
        self.subtask_attention_encoder=nn.Sequential(nn.Linear(self.len_task_repr,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,64),nn.Sigmoid())
        self.altruistic_subtask_attention_encoder=nn.Sequential(nn.Linear(self.len_task_repr,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,64),nn.Sigmoid())
        
        if use_others_obs:
            self.subtask_altruistic_encoder_v=nn.Sequential(nn.Linear(self.len_task_repr,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,self.modulation_dim*4))
            self.subtask_altruistic_encoder_p=nn.Sequential(nn.Linear(self.len_task_repr,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,self.modulation_dim*4))
        
        self.output_net_v1=nn.Sequential(nn.Linear(self.modulation_dim,self.modulation_dim),nn.ReLU(),nn.Linear(self.modulation_dim,self.modulation_dim))
        self.output_net_p1=nn.Sequential(nn.Linear(self.modulation_dim,self.modulation_dim),nn.ReLU(),nn.Linear(self.modulation_dim,self.modulation_dim))
        self.output_net_v2=nn.Sequential(nn.Linear(self.modulation_dim,self.modulation_dim),nn.ReLU(),nn.Linear(self.modulation_dim,1))
        self.output_net_p2=nn.Linear(self.modulation_dim,24)
        
        self.training_mode=False
        self.device="cuda" if self.cfg.use_gpu else "cpu"
        if self.training_stage in['sast','cooperation','perception','samt',"evaluate"]:
            self.epsilon_greedy=cfg.epsilon_greedy
        elif self.training_stage in ["mast-reachability","mast-feasibility"]:
            self.epsilon_greedy=1e-2
        else:
            raise NotImplementedError
    
        self.optimizer=torch.optim.Adam(self.parameters(),lr=self.lr)
        
        
        self.aux_names=["feas","reach","c2go","perc"]
        self.aux_terrain_cnn_encoder=nn.ModuleDict({a:nn.Sequential(nn.LayerNorm([1,8,8]),
                                        nn.Conv2d(1,32,kernel_size=5,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                        nn.Flatten())   for a in self.aux_names})
        self.aux_obj_cnn_encoder=nn.ModuleDict({a:nn.Sequential(nn.LayerNorm([20,8,8]),
                                        nn.Conv2d(20,32,kernel_size=5,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                        nn.Flatten())   for a in self.aux_names})
        self.aux_obj_attention=nn.ModuleDict({a:nn.Sequential(nn.Linear(self.len_task_repr,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,self.num_objects),nn.Sigmoid()) for a in self.aux_names})
        self.aux_terrain_mlp_encoder=nn.ModuleDict({a:nn.Linear(2304,128) for a in self.aux_names})
        self.aux_obj_mlp_encoder=nn.ModuleDict({a:nn.Linear(2304,128) for a in self.aux_names})
        self.aux_subtask_encoder=nn.ModuleDict({a:nn.Sequential(nn.Linear(self.len_task_repr,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,self.modulation_dim*2)) for a in self.aux_names})
        self.aux_recipe_encoder=nn.ModuleDict({a:nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,32)) for a in self.aux_names})
        self.aux_agent_encoder=nn.ModuleDict({a:nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,32)) for a in self.aux_names})
        self.aux_predictor=nn.ModuleDict({a:nn.Sequential(nn.Linear(self.modulation_dim,self.modulation_dim),nn.Dropout(),nn.ReLU(),nn.Linear(self.modulation_dim,self.modulation_dim),nn.Dropout(),nn.ReLU(),nn.Linear(self.modulation_dim,1)) for a in self.aux_names})
        self.aux_optimizer=torch.optim.Adam((v for k,v in self.named_parameters() if k.startswith("aux")),lr=self.lr)
        
        
    def state_encoder(self,obs,agent_idx):
        obs_map=obs["map"][:,agent_idx,...]
        obs_subtask=obs["subtask"][:,agent_idx,...]
        obs_agent=obs["agent"][:,agent_idx,...]
        obs_recipe=obs["recipes"][:,agent_idx,...]
        
        # encode map state
        obs_terrain=obs_map[:,:1,...]
        obs_obj=obs_map[:,1:,...]
        
        terrain_embed=self.cnn_encoder1(obs_terrain)
        terrain_embed_v=self.terrain_encoder_v(terrain_embed)
        terrain_embed_p=self.terrain_encoder_p(terrain_embed)
        
        subtask_attention_self=self.subtask_attention_encoder(obs_subtask)
        if self.num_agents>=2:
            obs_other_subtask=obs_subtask
            empty_subtask_mask=obs["subtask_aux"][:,agent_idx,...]  # shape: [bs,1]; 1 denotes altruistic subtask
            subtask_attention_altruistic=self.altruistic_subtask_attention_encoder(obs_other_subtask)
            subtask_attention=subtask_attention_altruistic*empty_subtask_mask+subtask_attention_self*(1-empty_subtask_mask)
        else:
            subtask_attention=subtask_attention_self
            
        subtask_attention=subtask_attention.unsqueeze(-1).unsqueeze(-1)
        obs_obj=self.cnn_encoder_before_attention(obs_obj)
        obs_obj=subtask_attention*obs_obj
        obj_embed=self.cnn_encoder11(obs_obj)
        obj_embed_v=self.obj_encoder_v(obj_embed)
        obj_embed_p=self.obj_encoder_p(obj_embed)
        
        map_embed_v=torch.cat([terrain_embed_v,obj_embed_v],dim=1)
        map_embed_p=torch.cat([terrain_embed_p,obj_embed_p],dim=1)
        
        # encode subtask
        subtask_embed_v=self.subtask_encoder_v(obs_subtask)
        subtask_embed_p=self.subtask_encoder_p(obs_subtask)
        
        if self.num_agents>=2:
            subtask_embed_p=self.subtask_altruistic_encoder_p(obs_other_subtask)*empty_subtask_mask+subtask_embed_p*(1-empty_subtask_mask)
            subtask_embed_v=self.subtask_altruistic_encoder_v(obs_other_subtask)*empty_subtask_mask+subtask_embed_v*(1-empty_subtask_mask)

        # encode agent 
        agent_embed_v=self.agent_encoder_v(obs_agent)
        agent_embed_p=self.agent_encoder_p(obs_agent)
        
        # encode recipe
        recipe_embed_v=self.recipe_encoder_v(obs_recipe)
        recipe_embed_p=self.recipe_encoder_p(obs_recipe)
            
        return torch.cat([map_embed_v,agent_embed_v,recipe_embed_v],dim=-1),subtask_embed_v,torch.cat([map_embed_p,agent_embed_p,recipe_embed_p],dim=-1),subtask_embed_p

    def set_training_mode(self, training_mode):
        self.training_mode=training_mode
        self.train() if training_mode else self.eval()
            
    def forward(self,obs,agent_idx=-1):
        state_feature_v,subtask_feature_v,state_feature_p,subtask_feature_p=self.state_encoder(obs,agent_idx)
        values=self.output_net_v1(state_feature_v*subtask_feature_v[:,:self.modulation_dim]+subtask_feature_v[:,self.modulation_dim:self.modulation_dim*2])
        values=self.output_net_v2(values*subtask_feature_v[:,self.modulation_dim*2:self.modulation_dim*3]+subtask_feature_v[:,self.modulation_dim*3:self.modulation_dim*4]).squeeze(-1)
        pi=self.output_net_p1(state_feature_p*subtask_feature_p[:,:self.modulation_dim]+subtask_feature_p[:,self.modulation_dim:self.modulation_dim*2])
        pi=self.output_net_p2(pi*subtask_feature_p[:,self.modulation_dim*2:self.modulation_dim*3]+subtask_feature_p[:,self.modulation_dim*3:self.modulation_dim*4])
        masks=obs["action_mask"][:,agent_idx,...].bool()
        dist=CategoricalMasked(logits=pi,epsilon_greedy=self.epsilon_greedy,mask=masks)
        actions = dist.sample()
        log_probs=torch.log(dist.probs[torch.arange(0,actions.shape[0]),actions])
        
        return actions,values,log_probs
    
    
    def predict_values(self,obs,agent_idx=-1):
        assert agent_idx>=0
        state_feature_v,subtask_feature_v,_,_=self.state_encoder(obs,agent_idx)
        values=self.output_net_v1(state_feature_v*subtask_feature_v[:,:self.modulation_dim]+subtask_feature_v[:,self.modulation_dim:self.modulation_dim*2])
        values=self.output_net_v2(values*subtask_feature_v[:,self.modulation_dim*2:self.modulation_dim*3]+subtask_feature_v[:,self.modulation_dim*3:self.modulation_dim*4]).squeeze(-1)
        
        return values
    
    def predict_aux(self,aux:str,obs,agent_idx):
        obs_map=obs["map"][:,agent_idx,...]
        obs_subtask=obs["subtask"][:,agent_idx,...]
        obs_agent=obs["agent"][:,agent_idx,...]
        obs_recipe=obs["recipes"][:,agent_idx,...]
        
        # encode map
        obs_terrain=obs_map[:,:1,...]
        obs_obj=obs_map[:,1:,...]
        
        obj_attention=self.aux_obj_attention[aux](obs_subtask).unsqueeze(-1).unsqueeze(-1)
        obs_obj=obj_attention*obs_obj
        
        terrain_embed=self.aux_terrain_cnn_encoder[aux](obs_terrain)
        terrain_embed=self.aux_terrain_mlp_encoder[aux](terrain_embed)
        
        obj_embed=self.aux_obj_cnn_encoder[aux](obs_obj)
        obj_embed=self.aux_obj_mlp_encoder[aux](obj_embed)
        
        map_embed=torch.cat([terrain_embed,obj_embed],dim=1)
        
        # encode subtask
        subtask_embed=self.aux_subtask_encoder[aux](obs_subtask)

        # encode agent 
        agent_embed=self.aux_agent_encoder[aux](obs_agent)
        
        # encode recipe
        recipe_embed=self.aux_recipe_encoder[aux](obs_recipe)
        
        state_embed=torch.cat([map_embed,agent_embed,recipe_embed],dim=1)
        
        aux_pred=self.aux_predictor[aux](state_embed*subtask_embed[:,:self.modulation_dim]+subtask_embed[:,self.modulation_dim:])
        
        if aux in ["feas","reach","perc"]:
            aux_pred=nn.Sigmoid()(aux_pred)
        
        return aux_pred
    
    def predict_aux_pss(self,aux:str,obs,pss,agent_idx):
        num_pss=pss.shape[0]
        obs_map=obs["map"][:,agent_idx,...]
        obs_agent=obs["agent"][:,agent_idx,...]
        obs_recipe=obs["recipes"][:,agent_idx,...]
        
        # encode map
        obs_terrain=obs_map[:,:1,...]
        obs_obj=obs_map[:,1:,...]
        
        obj_attention=self.aux_obj_attention[aux](pss).unsqueeze(-1).unsqueeze(-1)
        obs_obj=obj_attention*obs_obj
        
        terrain_embed=self.aux_terrain_cnn_encoder[aux](obs_terrain)
        terrain_embed=self.aux_terrain_mlp_encoder[aux](terrain_embed)
        terrain_embed=terrain_embed.repeat(num_pss,1)
        
        obj_embed=self.aux_obj_cnn_encoder[aux](obs_obj)
        obj_embed=self.aux_obj_mlp_encoder[aux](obj_embed)
        
        map_embed=torch.cat([terrain_embed,obj_embed],dim=1)
        

        # encode agent 
        agent_embed=self.aux_agent_encoder[aux](obs_agent).repeat(num_pss,1)
        
        # encode recipe
        recipe_embed=self.aux_recipe_encoder[aux](obs_recipe).repeat(num_pss,1)
        
        # encode subtask
        subtask_embed=self.aux_subtask_encoder[aux](pss)
        
        state_embed=torch.cat([map_embed,agent_embed,recipe_embed],dim=1)
        
        aux_pred=self.aux_predictor[aux](state_embed*subtask_embed[:,:self.modulation_dim]+subtask_embed[:,self.modulation_dim:])
        
        if aux in ["feas","reach","perc"]:
            aux_pred=nn.Sigmoid()(aux_pred)
        
        return aux_pred
    
    
    def evaluate_actions(self,obs,actions,agent_idx=-1):
        state_feature_v,subtask_feature_v,state_feature_p,subtask_feature_p=self.state_encoder(obs,agent_idx)
        values=self.output_net_v1(state_feature_v*subtask_feature_v[:,:self.modulation_dim]+subtask_feature_v[:,self.modulation_dim:self.modulation_dim*2])
        values=self.output_net_v2(values*subtask_feature_v[:,self.modulation_dim*2:self.modulation_dim*3]+subtask_feature_v[:,self.modulation_dim*3:self.modulation_dim*4]).squeeze(-1)
        pi=self.output_net_p1(state_feature_p*subtask_feature_p[:,:self.modulation_dim]+subtask_feature_p[:,self.modulation_dim:self.modulation_dim*2])
        pi=self.output_net_p2(pi*subtask_feature_p[:,self.modulation_dim*2:self.modulation_dim*3]+subtask_feature_p[:,self.modulation_dim*3:self.modulation_dim*4])
        masks=obs["action_mask"][:,agent_idx,...].bool()
        dist=CategoricalMasked(logits=pi,epsilon_greedy=self.epsilon_greedy,mask=masks)
        actions=actions[:,agent_idx].long()
        log_probs=torch.log(dist.probs[torch.arange(0,actions.shape[0]),actions.long()])
        entropies=dist.entropy()
        return values,log_probs,entropies
       
    
class TaskAllocator():
    def __init__(self,num_agents,last_subtask=None,t=None,len_task_repr=20):
        self.len_task_repr=len_task_repr
        self.num_agents=num_agents
        self.t=0 if t is None else t
        self.last_subtask=None if last_subtask is None else last_subtask
        self.empty_subtask=torch.zeros(len_task_repr,device="cuda")
        self.empty_subtask[0]=self.empty_subtask[10]=1
        self.last_pss=None
        self.last_matching=None
    
    def reset(self):
        self.last_subtask=None
        self.t=0
        
    def allocate(self,pss,feasibilities,reachabilities,c2gos):
        feasibilities=torch.clamp(feasibilities,0.1,0.9)
        feasibilities_all,_=torch.max(feasibilities,dim=1)
        reachabilities=torch.clamp(reachabilities,0.1,0.9)
        num_pss=pss.shape[0]
        num_agents=self.num_agents
        cost_matrix=torch.zeros(num_agents,max(num_pss*2,num_agents),device="cuda")
        for agent_idx in range(num_agents):
            for subtask_idx in range(num_pss):
                cost_matrix[agent_idx,subtask_idx]=-1*torch.log(feasibilities[subtask_idx][agent_idx])+(-20)*torch.log(reachabilities[subtask_idx][agent_idx])+0.2*c2gos[subtask_idx][agent_idx]
            for subtask_idx in range(num_pss,num_pss*2):
                cost_matrix[agent_idx,subtask_idx]=-1*torch.log(feasibilities_all[subtask_idx-num_pss])+0.2*c2gos[subtask_idx-num_pss][agent_idx]+60
        if self.last_pss is not None and torch.equal(pss,self.last_pss):
            if self.t>=20:
                for agent_idx in range(num_agents):
                    for subtask_idx in range(num_pss*2):
                        if self.last_matching[agent_idx]==subtask_idx:
                            cost_matrix[agent_idx,subtask_idx]+=200
                self.t=0
            else:
                for agent_idx in range(num_agents):
                    for subtask_idx in range(num_pss*2):
                        if self.last_matching[agent_idx]==subtask_idx:
                            cost_matrix[agent_idx,subtask_idx]-=200
        elif self.last_pss is not None and not torch.equal(pss,self.last_pss):
            self.t=0        
        
        if num_pss*2<self.num_agents:
            cost_matrix[:,num_pss*2:]=1000
            
        best_idxs = mnkrs.compute(cost_matrix.cpu().numpy().tolist())
        
        # coopeartion only
        allocation_id=[min(best_idxs[i][1],num_pss*2-1) for i in range(num_agents)]
        best_idxs=torch.LongTensor([min(best_idxs[i][1],num_pss*2-1) for i in range(num_agents)])
        
        subtasks_obs=torch.zeros((self.num_agents,self.len_task_repr),device="cuda")
        subtasks_aux_obs=torch.zeros((self.num_agents,1),device="cuda")
        for agent_idx in range(self.num_agents):
            if best_idxs[agent_idx]<num_pss:
                subtasks_obs[agent_idx]=pss[best_idxs[agent_idx]]
                subtasks_aux_obs[agent_idx]=0
            else:
                subtasks_obs[agent_idx]=pss[best_idxs[agent_idx]-num_pss]
                subtasks_aux_obs[agent_idx]=1
        
        self.t+=1
        self.last_pss=pss
        self.last_matching=best_idxs
        return subtasks_obs,subtasks_aux_obs,allocation_id
            

# adapted from openai stable baselines3 https://github.com/DLR-RM/stable-baselines3
class PPO():
    def __init__(
        self,
        env,
        eval_env,
        log_dir,
        cfg_file,
        **kwargs
    ):
        
        self.log_dir=log_dir
        
        self.cfg_file=cfg_file
        with open(self.cfg_file) as f:
            try:
                self.cfg = AttrDict(yaml.load(f, Loader=yaml.SafeLoader))
            except yaml.YAMLError as e:
                print(e)
        
        self.cfg.update(kwargs)
        
        self.num_agents=self.cfg.num_agents
        
        self.device = "cuda" if self.cfg.use_gpu else "cpu"
        
        self.env = env
        self.observation_space = None 
        self.action_space = spaces.Discrete
        self.num_timesteps = 0
        self.n_epochs=self.cfg.n_epochs
        
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.eval_env = eval_env
        self.policy=Policy_Overcooked(self.cfg).to(self.device)
        self.lr_scheduler_policy = torch.optim.lr_scheduler.StepLR(self.policy.optimizer, step_size=1,gamma=0.9,)
        self.lr_scheduler_policy_aux=torch.optim.lr_scheduler.StepLR(self.policy.aux_optimizer, step_size=1,gamma=0.9,)
        self._last_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        # When using VecNormalize:
        self._last_original_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._episode_num = 0
        
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1
        
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        self.training_stage=self.cfg.training_stage
        
        self.n_steps = self.cfg.n_steps
        
        # supervised learning, allow larger rollout length
        if self.training_stage in ["mast-reachability","mast-feasibility"]:
            self.n_steps*=4
        self.gamma = self.cfg.gamma
        self.gae_lambda = self.cfg.gae_lambda
        self.ent_coef = self.cfg.ent_coef
        self.vf_coef = self.cfg.vf_coef
        self.max_grad_norm = self.cfg.max_grad_norm
        self.rollout_buffer = None
        self.n_envs = self.cfg.num_envs

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
            self.cfg.batch_size > 1
        ), "`batch_size` must be greater than 1."
        
        self.batch_size = self.cfg.batch_size
        self.batch_size_aux=self.batch_size*16
        self.clip_range = self.cfg.clip_range
        self.clip_range_vf = lambda x: torch.clip(x, -self.cfg.clip_range_vf, self.cfg.clip_range_vf)
        self.target_kl = self.cfg.target_kl

        self.gradient_step=0
        
        self._setup_model()
            
        if self.cfg.state_dict_path is not None:
            self.load_state_dict(self.cfg.state_dict_path)   
        
        
        self.success_rate=0
        
        # curriculum learning; start from easy tasks to hard tasks
        for _ in range(5):
            self.env.goto_next_stage()
            self.eval_env.goto_next_stage()
            
        self.task_allocator=TaskAllocator(self.num_agents)
        

    def _setup_model(self) -> None:
        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            n_agents=self.num_agents,
            obs_shape=self.env.get_obs_shape(),
            action_dim=self.cfg.action_dim
        )
        
    def set_training_mode(self,training_mode):
        self.policy.set_training_mode(training_mode)
            
    def load_state_dict(self,path):
        print("loading state dict")
        checkpoint=torch.load(path)
        self.policy.load_state_dict(checkpoint["model_state_dict"],strict=False)
        
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        
        # Switch to train mode (this affects batch norm / dropout)
        self.set_training_mode(True)
        # Compute current clip range
        clip_range = self.clip_range
        # Optional: clip range for the value function

        entropy_losses = []
        pg_losses, value_losses = [], []
        
        continue_training = True
        # train policy networks for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for data_idx,rollout_data in enumerate(self.rollout_buffer.get(self.batch_size)):
                # if self.use_value_norm:
                #     self.value_normalizer.update(rollout_data.returns)
                actions = rollout_data.actions

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long()
                    
                values=[]
                log_probs=[]
                entropies=[]
                for idx in range(self.num_agents):
                    value, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions,agent_idx=idx)
                    values.append(self.clip_range_vf(value))
                    log_probs.append(log_prob)
                    entropies.append(entropy)
                    
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                returns=rollout_data.returns
            
                loss_sum=0
                for idx in range(self.num_agents):
                    # used for auxilary function learning
                    if self.training_stage in ["sast","perception","samt"] and idx>0:
                        continue
                    # ratio between old and new policy, should be one at the first iteration
                    ratio = torch.exp(log_probs[idx] - rollout_data.old_log_prob[:,idx])
                    policy_loss_1 = advantages[:,idx] * ratio
                    policy_loss_2 = advantages[:,idx] * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = torch.mean(-torch.min(policy_loss_1, policy_loss_2))
                    pg_losses.append(policy_loss.item())
                    
                    
                    # Value loss using the TD(gae_lambda) target
                    value_loss = torch.mean((returns[:,idx]-values[idx])**2)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    # # Approximate entropy when no analytical form
                    # entropy_loss = -torch.mean(-log_prob)
                    entropy_loss = -torch.mean(entropies[idx])

                    entropy_losses.append(entropy_loss.item())
                        
                    # ae_loss=self.policy.AE_forward(observations)
                    
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss # + goal_embed_loss*5e-3 # + ae_loss
                        
                    loss_sum+=loss
                    with torch.no_grad():
                        log_ratio = log_probs[idx] - rollout_data.old_log_prob[:,idx]
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.5f}")
                    break
                
                
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss_sum.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm,norm_type=inf)
                self.policy.optimizer.step()
                    
                self.gradient_step+=1
                    
            
            if not continue_training:
                break

        self._n_updates += self.n_epochs
    
    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)    
        print("current progress remaining:", self._current_progress_remaining)
    
    def _get_agent_action(self,actions):
            action_dict={}
            for i in range(self.num_agents):
                agent_name="agent-{}".format(i+1)
                action_dict[agent_name]=actions[i]
            return action_dict
        
    def collect_rollouts(
        self,
        env,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        print("collecting rollouts begins")
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        
        episode_cnts=0
            
        self._last_obs = self.env.reset() # pytype: disable=annotation-type-mismatch
                
        self._last_episode_starts = np.ones((self.n_envs,self.num_agents), dtype=bool)   
        while n_steps < n_rollout_steps:
            actions=[]
            values=[]
            log_probs=[]
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = self._last_obs
                for idx in range(self.num_agents):
                    action, value, log_prob=self.policy.forward(obs_tensor,agent_idx=idx)
                    if self.training_stage in ["sast","perception","samt","mast-reachability"] and idx>0:
                        action=torch.randint(0,4,(self.n_envs,),device=self.device)
                    actions.append(action)
                    values.append(value)
                    log_probs.append(log_prob)
            actions=torch.stack(actions).transpose(1,0).contiguous()
            values=torch.stack(values).transpose(1,0).contiguous()
            log_probs=torch.stack(log_probs).transpose(1,0).contiguous()
            
            actions_all=list(map(self._get_agent_action,actions.cpu().numpy().tolist()))
            new_obs, rewards, dones, is_terminals,_= env.step(actions_all)
            
            episode_cnts+=torch.sum((is_terminals*rewards)[:,0]>0)
            self.num_timesteps += self.n_envs
                
            n_steps += 1

            terminal_values=[]
            
            with torch.no_grad():
                for idx in range(self.num_agents):
                    terminal_value = self.policy.predict_values(new_obs,agent_idx=idx).cpu().flatten()
                    
                    # altruistic reward
                    if self.training_stage=="cooperation":
                        if idx==1:
                            last_reachability=self.policy.predict_aux("reach",self._last_obs,agent_idx=0)
                            new_reachability=self.policy.predict_aux("reach",new_obs,agent_idx=0)
                            extra_reward=torch.clamp((new_reachability-last_reachability)*torch.abs(new_reachability-last_reachability),-1,1).detach()*0.2
                            rewards[:,1:]+=extra_reward
                        
                    terminal_values.append(terminal_value)
            terminal_values=torch.stack(terminal_values).transpose(1,0)
            
            # Not used in 2cc situation because timeout is viewed as disability of completing subtask
            # rewards += (self.gamma * terminal_values * (1-is_terminals)*dones)
            
            
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs,dones)
            self._last_obs = new_obs
            self._last_episode_starts = dones.cpu().numpy()
            
            if dones.any():
                self._last_obs=env.reset_terminated(dones)

        terminal_values=[]
        with torch.no_grad():
            # compute value for the last observation
            terminal_value = self.policy.predict_values(new_obs,agent_idx=idx).cpu().flatten()
            terminal_values.append(terminal_value)
        terminal_values=torch.stack(terminal_values).transpose(1,0)

        rollout_buffer.compute_returns_and_advantage(last_values=terminal_values, dones=dones)
        average_score=episode_cnts/(self.n_envs*self.n_steps)*512
        
        
        
        # curriculum learning in rl training
        # if average_score>50:
        #     self.env.goto_next_stage()
        #     self.eval_env.goto_next_stage()
        return None
    
    
    # self-imitation learning, faster than pure rl
    def train_si(self):
        self.set_training_mode(True)
        losses=[]
        for rollout_data in self.rollout_buffer.get_sil_batch(self.batch_size):
            actions = rollout_data.actions
            if actions.nelement()==0:
                return 

            if isinstance(self.action_space, spaces.Discrete):
                actions = rollout_data.actions.long()
                
            loss=0
            for idx in range(self.num_agents):
                # Used when we are not training another agent
                if self.training_stage in ["sast","samt"] and idx>0:
                    break
                _, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, rollout_data.actions,agent_idx=idx)
                bc_loss= -log_prob
                loss+=torch.mean(bc_loss)*self.cfg.sl_coef
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())
        
            
    # train the cost-to-go function
    def train_distance_to_goal(self):
        self.set_training_mode(True)
        losses=[]
        for rollout_data in self.rollout_buffer.get_distance_pair(self.batch_size_aux):
            target_distance_to_goal=rollout_data.distance_to_goal
            target_distance_to_goal=target_distance_to_goal[:,:1]
            loss=0
            distance= self.policy.predict_aux("c2go",rollout_data.observations,agent_idx=0)
            invalid_mask=(target_distance_to_goal>800).float()
            if torch.sum(1-invalid_mask)<1:
                continue
            loss=torch.sum((distance+10-target_distance_to_goal)**2*(1-invalid_mask))/torch.sum(1-invalid_mask)*1e-4
            self.policy.aux_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.aux_optimizer.step()
            losses.append(loss.item())
        print({"distance_loss":sum(losses)/len(losses)})
        
    # train the reachability function
    def train_reachability(self):
        self.set_training_mode(True)
        losses=[]
        accuracies=[]
        for rollout_data in self.rollout_buffer.get_distance_pair(self.batch_size_aux):
            target_distance_to_goal=rollout_data.distance_to_goal
            loss=0
            reachabilitys=[]
            for idx in range(self.num_agents):
                reachability= self.policy.predict_aux("reach",rollout_data.observations,agent_idx=idx)
                reachabilitys.append(reachability)
            reachabilitys=torch.stack(reachabilitys).transpose(1,0).squeeze(-1)
            reachabilitys=torch.clamp(reachabilitys,1e-3,1-1e-3)
            reachability_gt=(target_distance_to_goal<800).float()
            invalid_mask=(target_distance_to_goal>1e5-3).float()
            invalid_mask[:,1:]=1   # mask empty task
            num_positive=torch.sum((reachability_gt==1)*(1-invalid_mask))
            num_negative=torch.sum((reachability_gt==0)*(1-invalid_mask))
            
            if num_negative>num_positive:
                neg_index=((torch.round(reachability_gt)==0)*(1-invalid_mask)).nonzero()[:(num_negative-num_positive).long()]
                invalid_mask[neg_index]=1
            else:
                print("Number of negative samples is smaller than positive samples")
            if torch.sum(1-invalid_mask)<1:
                continue
            accuracy=(torch.sum((torch.round(reachabilitys)==reachability_gt)*(1-invalid_mask))/torch.sum(1-invalid_mask)).item()
            loss=-(reachability_gt*torch.log(reachabilitys)+(1-reachability_gt)*torch.log(1-reachabilitys))*(1-invalid_mask)
            loss=torch.sum(loss)/(self.batch_size_aux*2-torch.sum(invalid_mask))*1e-2
            self.policy.aux_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.aux_optimizer.step()
            losses.append(loss.item())
            accuracies.append(accuracy)
        print({"reachability loss":sum(losses)/len(losses)})
        print({"reachability accuracy":sum(accuracies)/len(accuracies)})


        # two consecutive states should have similar reachability, make the reachability function more robust, less interruptable
        self.set_training_mode(True)
        for rollout_data_1,rollout_data_2 in self.rollout_buffer.get_consecutive_pair(self.batch_size_aux):
            reachability_1= self.policy.predict_aux("reach",rollout_data_1.observations,agent_idx=0)
            reachability_2= self.policy.predict_aux("reach",rollout_data_2.observations,agent_idx=0)
            valid_flag=(1-rollout_data_2.episode_starts.float())[:,0:1]
            loss=valid_flag*(reachability_1-reachability_2)**2
            loss=torch.sum(loss)/torch.sum(valid_flag)*1e-1
            self.policy.aux_optimizer.zero_grad()
            loss.backward()
            self.policy.aux_optimizer.step()
        print(f"consecutive similarity loss: {loss.item()}")
            
        
    # train the feasibility function
    def train_feasibility(self):
        self.set_training_mode(True)
        losses=[]
        accuracies=[]
        for _ in range(5):
            for rollout_data in self.rollout_buffer.get_distance_pair(self.batch_size_aux):
                target_distance_to_goal=rollout_data.distance_to_goal
                loss=0
                feasibilities=[]
                for idx in range(self.num_agents):
                    feasibility= self.policy.predict_aux("feas",rollout_data.observations,agent_idx=idx)
                    feasibilities.append(feasibility)
                feasibilities=torch.stack(feasibilities).transpose(1,0).squeeze(-1)
                feasibilities=torch.clamp(feasibilities,1e-3,1-1e-3)
                feasibility_gt=(target_distance_to_goal<800).float()
                invalid_mask=(target_distance_to_goal>1e5-3).float()
                invalid_mask[:,1:]=1   # mask empty task
                num_positive=torch.sum((feasibility_gt==1)*(1-invalid_mask))
                num_negative=torch.sum((feasibility_gt==0)*(1-invalid_mask))
                
                print(num_positive,num_negative)
                if num_negative>num_positive:
                    neg_index=((torch.round(feasibility_gt)==0)*(1-invalid_mask)).nonzero()[:(num_negative-num_positive).long()]
                    invalid_mask[neg_index]=1
                else:
                    print("Number of negative samples is smaller than positive samples")
                accuracy=(torch.sum((torch.round(feasibilities)==feasibility_gt)*(1-invalid_mask))/torch.sum(1-invalid_mask)).item()
                loss=-(feasibility_gt*torch.log(feasibilities)+(1-feasibility_gt)*torch.log(1-feasibilities))*(1-invalid_mask)
                loss=torch.sum(loss)/(self.batch_size_aux*2-torch.sum(invalid_mask))*1e-2
                self.policy.aux_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.aux_optimizer.step()
                losses.append(loss.item())
                accuracies.append(accuracy)
        
        self.set_training_mode(True)
        for rollout_data_1,rollout_data_2 in self.rollout_buffer.get_consecutive_pair(self.batch_size_aux):
            reachability_1= self.policy.predict_aux("feas",rollout_data_1.observations,agent_idx=0)
            reachability_2= self.policy.predict_aux("feas",rollout_data_2.observations,agent_idx=0)
            valid_flag=(1-rollout_data_2.episode_starts.float())[:,0:1]
            loss=valid_flag*(reachability_1-reachability_2)**2
            loss=torch.sum(loss)/torch.sum(valid_flag)
            self.policy.aux_optimizer.zero_grad()
            loss.backward()
            self.policy.aux_optimizer.step()
        
    # train the perception function
    def train_perception(self):
        acc_count=0
        best_acc=0
        for epochs in range(10000):
            self.set_training_mode(True)
            losses=0
            correct=0
            total=0
            for _ in range(128):
                obs = self.env.reset()
                for __ in range(random.randint(0,20)):
                    obs=self.env.random_step()
                perception_gt=torch.from_numpy(self.env.compute_perception_gt()).float().to("cuda")
                perception_pred=self.policy.predict_aux(aux="perc",obs=obs,agent_idx=0)[:,0]
                losses+=nn.BCELoss(reduction="sum")(perception_pred,perception_gt)
                correct+=(torch.abs(torch.round(perception_pred)-perception_gt)<0.01).sum().item()
                total+=self.n_envs

            self.policy.aux_optimizer.zero_grad()
            loss=losses/total
            loss.backward()
            self.policy.aux_optimizer.step()
            accuracy=correct/total
            
            if epochs%10==0:
                self.set_training_mode(False)
                losses=0
                correct=0
                total=0
                with torch.no_grad():
                    for _ in range(512):
                        obs = self.env.reset(eval=True)
                        perception_gt=torch.from_numpy(self.env.compute_perception_gt()).float().to("cuda")
                        perception_pred=self.policy.predict_aux("perc",obs,agent_idx=0)[:,0]
                        correct+=(torch.abs(torch.round(perception_pred)-perception_gt)<0.01).sum().item()
                        total+=self.n_envs

                accuracy=correct/total
                if accuracy>0.99:  # stop training if accuracy is high enough
                    acc_count+=1
                if accuracy>0.95:
                    self.lr_scheduler_policy_aux.step()
                if accuracy>best_acc:
                    best_acc=accuracy
                    os.makedirs(os.path.join(self.log_dir,'state_dicts'),exist_ok=True)
                    state_dict={}
                    state_dict['model_state_dict']=self.policy.state_dict()
                    state_dict['optimizer_state_dict']=self.policy.optimizer.state_dict()
                    torch.save(state_dict,os.path.join(self.log_dir,'state_dicts','{:.4f}.pth'.format(accuracy*100)))
            if acc_count==5:
                break
        
    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        eval_env = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":
        # torch.autograd.set_detect_anomaly(True)
        self.iteration = 0

        self.start_time = time.time()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = np.ones((self.n_envs,self.num_agents), dtype=bool)
        while self.num_timesteps < total_timesteps:
            if isinstance(self.action_space,spaces.Discrete):
                self.policy.epsilon_greedy*=0.98
            if not self.training_stage=="perception":
                self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps)
                self.rollout_buffer.process_distance_pair()

            self.iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if self.training_stage in ["sast","samt","cooperation"]:
                self.train()
                if self.training_stage not in ["cooperation"]:
                    self.train_si()
            if self.training_stage=="perception":
                self.train_perception()
            if self.training_stage=="mast-reachability":
                self.train_reachability()
            if self.training_stage=="mast-feasibility":
                self.train_distance_to_goal()
                self.train_feasibility()
            os.makedirs(os.path.join(self.log_dir,'state_dicts'),exist_ok=True)
            state_dict={}
            state_dict['model_state_dict']=self.policy.state_dict()
            state_dict['optimizer_state_dict']=self.policy.optimizer.state_dict()
            torch.save(state_dict,os.path.join(self.log_dir,'state_dicts','{}.pth'.format(self.num_timesteps)))
            if self.training_stage=="perception":
                exit(0)
            
        return self
    
    
    @torch.no_grad()
    def evaluate_mamt(self,samples):
        self.eval_env.set_testing_samples(samples)
        epsilon_backup=copy.deepcopy(self.policy.epsilon_greedy)
        self.policy.epsilon_greedy=0
        self.set_training_mode(False)
        self.task_allocator.reset()
        cnt=0
        cnts=[]
        obs=self.eval_env.reset(eval=True,derandom=True)
        frames=[]
        success=0
        failure=0
        score=0
        score_square=0
        cum_score=0
        sample_cnt=0
        # print("evaluating...")
        while True:
            branch_count=0
            while True:  # branching
                branch_count+=1
                pss_size=torch.sum(torch.sum(obs["pss"][0],dim=1)>0).item()
                pss=obs["pss"][0][:pss_size,...]
                if any(self.eval_env.is_branching()):
                    perception=self.policy.predict_aux_pss("perc",obs=obs,pss=pss,agent_idx=0)
                    done=self.eval_env.step_perception(perception[:,0])
                else:
                    break
                if branch_count>=5:
                    break
            obs=self.eval_env.get_repr()
            pss_size=torch.sum(torch.sum(obs["pss"][0],dim=1)>0).item()
            pss=obs["pss"][0][:pss_size,...]
            non_perception=(torch.sum(pss[:,7:10],dim=1)==0).nonzero().squeeze(1)
            if not non_perception.nelement()==0:
                pss=pss[non_perception,...]
                pss_size=pss.shape[0]
                exit_program=False
            else:
                exit_program=True
            
            feasibilities=[]
            reachabilities=[]
            c2gos=[]
            for agent_id in range(self.num_agents):
                feasibilities.append(self.policy.predict_aux_pss("feas",obs=obs,pss=pss,agent_idx=agent_id))
                reachabilities.append(self.policy.predict_aux_pss("reach",obs=obs,pss=pss,agent_idx=agent_id))
                c2gos.append(self.policy.predict_aux_pss("c2go",obs=obs,pss=pss,agent_idx=agent_id))
            feasibilities=torch.stack(feasibilities,dim=1)
            reachabilities=torch.stack(reachabilities,dim=1)
            c2gos=torch.stack(c2gos,dim=1)
            allocate_subtasks,allocate_subtasks_aux,allocation_id=self.task_allocator.allocate(pss,feasibilities,reachabilities,c2gos)
            obs["subtask"]=allocate_subtasks.unsqueeze(0)
            obs["subtask_aux"]=allocate_subtasks_aux.unsqueeze(0)
            actions=[]
            for idx in range(self.num_agents):
                action, _, _=self.policy.forward(obs,agent_idx=idx)
                if self.training_stage in ["sast","perception","samt","mast-reachability"] and idx>0:
                    action=torch.randint(0,4,(1,),device=self.device)
                actions.append(action)
            actions=torch.stack(actions).transpose(0,1).contiguous()
            actions=list(map(self._get_agent_action,actions.cpu().numpy().tolist()))
            obs, reward, done, terminated,_ = self.eval_env.step(actions)
            if exit_program:
                done=done | 1
                terminated=terminated | 1
            cum_score+=0.2*torch.max(reward[0]+0.01).item()*((0.99)**obs["timestep"][0][0].item())
            if done.any():
                sample_cnt+=1
                if terminated.any() or cum_score>=1.0:                     # since some task involves an infinite loop(such as while(onFire)PutOutFire), we use a score threshold to determine whether the task is finished
                    cum_score+=1*(0.99**obs["timestep"][0][0].item())      # finish task
                    success+=1
                else:
                    failure+=1
                score+=cum_score
                score_square+=cum_score**2
                cum_score=0
                if sample_cnt==len(samples):
                    break
                obs = self.eval_env.reset_terminated(dones=done,eval=True,derandom=True)
                
            # save the first demo
            if sample_cnt<=1:  
                for subtask_id in range(pss_size):
                    self.eval_env.add_annotation(f"reach_{subtask_id}",[reachabilities[subtask_id][0].item(),reachabilities[subtask_id][1].item()])
                    self.eval_env.add_annotation(f"feas_{subtask_id}",[feasibilities[subtask_id][0].item(),feasibilities[subtask_id][1].item()])
                self.eval_env.add_annotation("allocation",allocation_id)
                frame=self.eval_env.render()[0]
                frame=cv2.resize(frame,(1200,800))
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        self.policy.epsilon_greedy=epsilon_backup
        success_rate=success/(success+failure)
        average_score=score/(success+failure)
        std=math.sqrt((score_square/(success+failure))-score**2/(success+failure)**2)
        
        videoWriter=cv2.VideoWriter("demo.avi",cv2.VideoWriter_fourcc('I', '4', '2', '0'),1.0,(1200,800),True)
        for frame in frames:
            videoWriter.write(frame)
        videoWriter.release()
        return success_rate,average_score,std