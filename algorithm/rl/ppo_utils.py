from sys import setprofile
from tracemalloc import start
import numpy as np
import torch
from copy import deepcopy
from typing import Any, Dict, Optional, Type, Union, NamedTuple, List
import torch.nn as nn


class RolloutBufferSamples(NamedTuple):
    observations:Dict[str, torch.Tensor]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    episode_starts:torch.Tensor
    distance_to_goal: torch.Tensor
    

  
class RolloutBuffer():
    """
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        device,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        n_agents: int=1,
        obs_shape: Dict[str,tuple] = None,
        action_dim: int = 1,
    ):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = [],[],[],[]
        self.returns, self.episode_starts, self.values, self.log_probs = [],[],[],[]
        self.generator_ready = False
        self.n_agents=n_agents
        self.obs_shape=obs_shape
        self.action_dim=action_dim
        self.reset()

    def reset(self) -> None:
        self.observations = [[] for _ in range(self.n_envs)]
        for k,v in self.obs_shape.items():
            setattr(self,k,np.zeros((self.buffer_size,self.n_envs,self.n_agents)+v))
        
        self.actions = np.zeros((self.buffer_size, self.n_envs,self.n_agents,self.action_dim), dtype=np.float32)
        if self.action_dim==1:
            self.actions=self.actions.squeeze(-1)
        self.rewards = np.zeros((self.buffer_size, self.n_envs,self.n_agents), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs,self.n_agents), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs,self.n_agents), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs,self.n_agents), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs,self.n_agents), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs,self.n_agents), dtype=np.float32)
        self.t=np.zeros((self.buffer_size, self.n_envs,self.n_agents), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False
        self.distance_to_goal=np.zeros((self.buffer_size, self.n_envs,self.n_agents), dtype=np.float32)
        self.dones=np.zeros((self.buffer_size, self.n_envs,self.n_agents), dtype=np.float32)
        
        
        self._tensor_names = [
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "episode_starts",
            "distance_to_goal",
        ]+ list(self.obs_shape.keys())

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).

        """

        # Convert to numpy
        last_values = last_values.cpu().numpy().copy()
        dones=dones.cpu().numpy().copy()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            
        self.returns = self.advantages + self.values

    def add(
        self,
        obs,
        action: torch.Tensor,
        reward: torch.Tensor,
        episode_start: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        dones: torch.Tensor
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """ 
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)
        
        for k,_ in self.obs_shape.items():
            getattr(self,k)[self.pos]=np.array(obs[k].cpu()).copy()
            
        self.actions[self.pos] = np.array(action.cpu()).copy()
        self.rewards[self.pos] = np.array(reward.cpu()).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.cpu().numpy().copy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.dones[self.pos]=dones.clone().cpu().numpy().copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            
    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    
    def get(self, batch_size):
        data_size=self.buffer_size * self.n_envs
        indices = np.random.permutation(data_size)
        # Prepare the data
        if not self.generator_ready:
            if isinstance(self.observations[0],List):
                self.observations=sum(self.observations,[])
            for tensor in self._tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < data_size:
            if start_idx + batch_size >=data_size:
                end_idx = data_size
                yield self._get_samples(indices[start_idx : end_idx])
                return
            else:
                yield self._get_samples(indices[start_idx : start_idx + batch_size])
                start_idx += batch_size
                
    def get_prioritized_batch(self,batch_size):
        self.priority=torch.from_numpy()
        data_size=self.buffer_size * self.n_envs
        indices = torch.distributions.categorical.Categorical(probs=self.priority).sample((batch_size,))
        # Prepare the data
        if not self.generator_ready:
            if isinstance(self.observations[0],List):
                self.observations=sum(self.observations,[])
            for tensor in self._tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        return self._get_samples(indices)
    
    
    def unflatten(self,arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        return arr.reshape(self.n_envs,self.buffer_size, *shape[1:]).swapaxes(0, 1)
    
    
    def get_sil_batch(self,batch_size):
        for tensor in self._tensor_names:
            try:
                self.__dict__[tensor] = self.unflatten(self.__dict__[tensor])
            except:
                print("unflatten error!",tensor,self.__dict__[tensor].shape)
        self.generator_ready=False
        
        self.success_mask=(self.rewards>0)
        sliding_window=deepcopy(self.success_mask)
        for i in range(10):
            sliding_window=np.roll(sliding_window,-1,axis=0)*(1-np.roll(self.episode_starts,1,axis=0))
            self.success_mask=np.logical_or(self.success_mask,sliding_window)
        
        indices=np.where(self.success_mask[:,:,0].swapaxes(0,1).flatten())
        data_size=len(indices)
        # Prepare the data
        if not self.generator_ready:
            for tensor in self._tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
            
        start_idx = 0
        while start_idx < data_size:
            if start_idx + batch_size >=data_size:
                end_idx = data_size
                yield self._get_samples(indices[start_idx : end_idx])
                return
            else:
                yield self._get_samples(indices[start_idx : start_idx + batch_size])
                start_idx += batch_size
            
    def process_distance_pair(self):
        last_distance=np.ones((self.n_envs,self.n_agents))*1e5
        for ii in range(self.buffer_size-1,-1,-1):
            bool_subtask_completed=np.logical_and(self.dones[ii,:,:],self.rewards[ii,:,:]>0.5)  # agent itself complete a subtask
            bool_violate_program=np.logical_and(np.logical_and(self.dones[ii,:,:],self.rewards[ii,:,:]<0),self.rewards[ii,:,::-1]>-0.05)    # violate program or timeout, which means the task is difficult to complete
            bool_invalid=np.logical_and(self.dones[ii,:,:],np.logical_not(np.logical_or(bool_subtask_completed,bool_violate_program))) # other agent complete a task, thus the data becomes invalid
            
            self.distance_to_goal[ii,:,:]=(last_distance+1)*(1-self.dones[ii,:,:])+1*bool_subtask_completed+1000*bool_violate_program+1e5*bool_invalid
            last_distance=self.distance_to_goal[ii,:,:]
        return 
    
    
    def get_distance_pair(self,batch_size):
        if not self.generator_ready:
            if isinstance(self.observations[0],List):
                self.observations=sum(self.observations,[])
            for tensor in self._tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True
            
        data_size=self.buffer_size * self.n_envs
        indices = np.random.permutation(data_size)

        start_idx = 0
        while start_idx < data_size:
            if start_idx + batch_size >=data_size:
                end_idx = data_size
                yield self._get_samples(indices[start_idx : end_idx])
                return
            else:
                yield self._get_samples(indices[start_idx : start_idx + batch_size])
                start_idx += batch_size
                
    def get_consecutive_pair(self,batch_size):
        data_size=self.buffer_size * self.n_envs
        indices_1 = np.random.permutation(data_size)
        indices_1=np.where(indices_1%self.buffer_size==self.buffer_size-1,indices_1-1,indices_1)
        indices_2 = indices_1+1
        if not self.generator_ready:
            if isinstance(self.observations[0],List):
                self.observations=sum(self.observations,[])
            for tensor in self._tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        start_idx = 0
        while start_idx < data_size:
            if start_idx + batch_size >=data_size:
                end_idx = data_size
                yield (self._get_samples(indices_1[start_idx : end_idx]),self._get_samples(indices_2[start_idx:end_idx]))
                return
            else:
                yield (self._get_samples(indices_1[start_idx : start_idx + batch_size]),self._get_samples(indices_2[start_idx:start_idx+batch_size]))
                start_idx += batch_size
        pass
            
    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        try:
            if copy:
                if isinstance(array, np.ndarray):
                    return torch.from_numpy(array).float().to(self.device)
                elif isinstance(array, dict):
                    return {key: self.to_torch(value, copy) for key, value in array.items()}
                else:
                    raise NotImplementedError
        except:
            return array
    
    def _get_samples(self, batch_inds: np.ndarray):
        data = ({k:getattr(self,k)[batch_inds] for k in self.obs_shape.keys()},
            # [self.recipe_observations[batch_inds],self.command_observations[batch_inds],self.map_observations[batch_inds],self.action_mask_observations[batch_inds],self.action_history_observations[batch_inds],self.t[batch_inds],self.agent_state_observation[batch_inds]],
            self.actions[batch_inds],
            self.values[batch_inds],
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds],
            self.episode_starts[batch_inds],
            self.distance_to_goal[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
    
      
def get_schedule_fn(value):
    #TODO: Add support for other types of schedules
    return (lambda x: value)
