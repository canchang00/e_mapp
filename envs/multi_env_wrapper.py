import os
import multiprocessing
from collections import OrderedDict
from typing import Sequence

import gym
import numpy as np
import torch

def _worker(remote, parent_remote, env):
    parent_remote.close()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info,is_debut = env.step(data)
                remote.send((observation, reward, done, info,is_debut))
            elif cmd == 'reset':
                eval,derandom=data
                observation = env.reset(eval=eval,derandom=derandom)
                remote.send(observation)
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd=="goto_next_stage":
                env.goto_next_stage()
            elif cmd=='reset_terminated':
                done,eval,derandom=data
                if done:
                    remote.send(env.reset(eval=eval,derandom=derandom))
                else:
                    obs=env.get_repr()
                    remote.send(obs)
            elif cmd=='get_repr':
                obs=env.get_repr()
                remote.send(obs)
            elif cmd=='random_step':
                remote.send(env.random_step())
            elif cmd=='print_follow_program_stats':
                env.print_follow_program_stats()
            elif cmd=='compute_perception_gt':
                answer=env.compute_perception_gt()
                remote.send(answer)
            elif cmd=='render':
                remote.send(env.render(data))
            elif cmd=="set_testing_samples":
                remote.send(env.set_testing_samples(data))
            elif cmd=="add_annotation":
                anno_name,anno_content=data
                env.annotations[anno_name]=anno_content
            elif cmd=="is_branching":
                remote.send(env.is_branching())
            elif cmd=="step_perception":
                remote.send(env.step_perception(data))
            else:
                raise NotImplementedError("`{}` is not implemented in the worker".format(cmd))
        except EOFError:
            break


class SubprocVecEnv(object):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env,n_env, start_method=None):
        self.waiting = False
        self.closed = False
        self.n_envs = n_env
        self.env=env
        self.num_agents=env.num_agents
        self.device=self.env.device

        # In some cases (like on GitHub workflow machine when running tests),
        # "forkserver" method results in an "connection error" (probably due to mpi)
        # We allow to bypass the default start method if an environment variable
        # is specified by the user
        if start_method is None:
            start_method = os.environ.get("DEFAULT_START_METHOD")

        # No DEFAULT_START_METHOD was specified, start_method may still be None
        if start_method is None:
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(self.n_envs)])
        self.processes = []
        for work_remote, remote, env_ in zip(self.work_remotes, self.remotes, [env for i in range(n_env)]):
            args = (work_remote, remote, env_)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.obs_keys=self.env.get_obs_keys()
        
    def _to_tensor(self,x):
        if isinstance(list(x.values())[0], torch.Tensor):
            return x
        else:
            return {k:torch.from_numpy(v).float().to(self.device) for k,v in x.items()}

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_await(self):
        results = [list(remote.recv()) for remote in self.remotes]
        self.waiting = False
        obs, rewards, dones, is_terminals,is_debut = zip(*results)
        obs={key:np.stack([o[key] for o in obs]) for key in obs[0].keys()}
        is_terminals=torch.LongTensor(np.array(list(is_terminals))).to(self.device)
        rewards=torch.FloatTensor(np.array(list(rewards))).to(self.device)
        dones=torch.LongTensor(np.array(list(dones))).to(self.device)
        return self._to_tensor(obs),rewards,dones,is_terminals,is_debut

    def step(self,actions):
        self.step_async(actions)
        return self.step_await()

    def reset(self,eval=False,derandom=False):
        for remote in self.remotes:
            remote.send(('reset', (eval,derandom)))
        obs = [remote.recv() for remote in self.remotes]
        obs={key:np.stack([o[key] for o in obs]) for key in obs[0].keys()}
        # obs=list(zip(*obs))
        # obs=[np.stack(obs_i) for obs_i in obs]
        return self._to_tensor(obs)

    def reset_terminated(self,dones,eval=False,derandom=False):
        dones=dones[:,0]
        for idx,remote in enumerate(self.remotes):
            remote.send(('reset_terminated',(dones[idx].item(),eval,derandom)))
        obs=[remote.recv() for remote in self.remotes]
        obs={key:np.stack([o[key] for o in obs]) for key in obs[0].keys()}
        return self._to_tensor(obs)
    
    def compute_perception_gt(self):
        for idx,remote in enumerate(self.remotes):
            remote.send(('compute_perception_gt',None))
        answers=[remote.recv() for remote in self.remotes]
        answers=np.stack(answers)
        return answers

    def add_annotation(self,anno_name,anno_content):
        self.remotes[0].send(('add_annotation',(anno_name,anno_content)))
        return 
    
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True
        
    def goto_next_stage(self):
        for remote in self.remotes:
            remote.send(('goto_next_stage', None))
        return

    def print_follow_program_stats(self):
        for idx,remote in enumerate(self.remotes):
            remote.send(('print_follow_program_stats', None))
            
    def random_step(self):
        for idx,remote in enumerate(self.remotes):
            remote.send(('random_step', None))
        obs = [remote.recv() for remote in self.remotes]
        obs={key:np.stack([o[key] for o in obs]) for key in obs[0].keys()}
        return self._to_tensor(obs)
    
    def get_repr(self):
        for idx,remote in enumerate(self.remotes):
            remote.send(('get_repr', None))
        obs = [remote.recv() for remote in self.remotes]
        obs={key:np.stack([o[key] for o in obs]) for key in obs[0].keys()}
        return self._to_tensor(obs)
    
    def is_branching(self):
        for idx,remote in enumerate(self.remotes):
            remote.send(('is_branching', None))
        ret = [remote.recv() for remote in self.remotes]
        return ret
    
    def step_perception(self,perception):
        self.remotes[0].send(('step_perception',perception))
        return self.remotes[0].recv()
            
    def render(self,data=None):
        for idx,remote in enumerate(self.remotes):
            remote.send(('render',data))
        img=[remote.recv() for remote in self.remotes]
        return img
            
    def get_obs_shape(self):
        return self.env.get_obs_shape()

    def set_testing_samples(self,samples):
        assert self.n_envs==1
        self.remotes[0].send(('set_testing_samples',samples))
        self.remotes[0].recv()