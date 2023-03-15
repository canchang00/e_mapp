# Recipe planning
import threading
from unicodedata import name

# Other core modules
# from utils.interact import interact
from utils.world import World
from utils.core import *
from utils.agent import SimAgent
from misc.game.gameimage import GameImage
from utils.agent import COLORS
import copy
import networkx as nx
import numpy as np
from itertools import combinations, permutations, product
from collections import namedtuple,OrderedDict
import gym
from gym import error, spaces, utils
from envs.display import GameScreen
from algorithm.parser.program_executor import ProgramExecutor
from envs.env_utils import *
import random
import threading
import multiprocessing
import torch
import sys
from termcolor import colored as color
import os
import cv2

CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")



EPS=1e-7
ACTION_LIST=[[0,1],[1,0],[0,-1],[-1,0],[0,0]]
ACTION_TYPE=["move","pick","drop","interact","merge","serve"]

def get_agent_action(actions):
    if actions==0:
        actions_all={"agent-1":(0,1)}
    elif actions==1:
        actions_all={"agent-1":(1,0)}
    elif actions==2:
        actions_all={"agent-1":(0,-1)}
    elif actions==3:
        actions_all={"agent-1":(-1,0)}
    else:
        raise NotImplementedError
    return actions_all

# utility functions
def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]


def merge(x):
    has_plate=False
    ingredients=[]
    for ele in x:
        if ele.endswith('p'):
            ingredients.append(ele[:-1])
            has_plate=True
        elif ele.startswith('p'):
            ingredients.append(ele[1:])
            has_plate=True
        else:
            ingredients.append(ele)
    ingredients=list(set(ingredients))
    return ''.join(sorted(ingredients)) if not has_plate else 'p'+''.join(sorted(ingredients))
    
fullname2abbreviation={
    'Plate':'p',
    'FreshTomato':'Ft',
    'ChoppedTomato':'Ct',
    'FreshOnion':'Fo',
    'ChoppedOnion':'Co',
    'Dirtyplate':'q',
}
# type of functions
FUNCTION_TYPE={
    "Empty":0,
    "Pick":1,
    "Chop":2,
    "Merge":3,
    "Serve":4,
    "Wash":5,
    "PutOffFire":6,
    "Tautology":7,
    "CheckDemand":8,
    "IsOnFire":9,
}

# type of arguments
ARGUMENT_TYPE={
    "Empty":0,
    "Ft":1,
    "Fo":2,
    "Ct":3,
    "Co":4,
    "p":5,
    "pCt":6,
    "pCo":7,
    "CoCt":8,
    "pCoCt":9,
}

OBJ_TYPE={
    "E":0,
    "t":1,
    "o":2,
    "Ct":3,
    "Co":4,
    "p":5,
    "q":6,
    "pCt":7,
    "pCo":8,
    "pCoCt":9,
    "CoCt":10,
    "e":11,  # extinguisher
    "h":12,
    "v":13,
    "/":14,
    "*":15,
    "w":16,
    "T":17,
    "O":18,
    "agent-1":19, # agent-1
    "agent-2":20 # agent-2
}


legal_subtasks = [(1,1),(1,2),(1,3),(1,4),(1,5),\
                    (2,1),(2,2),\
                    (3,[3,4]),
                    (5,5)]
            
class subtask():
    def __init__(self,function_type,function_arg=[],lineno=-1):
        self.function_type=function_type
        self.lineno=lineno
        if isinstance(function_arg,list):
            self.function_arg=function_arg
        else:
            self.function_arg=[function_arg]
            
    def __repr__(self):
        return str(get_key(FUNCTION_TYPE,self.function_type)[0])+" "+'-'.join(str(get_key(ARGUMENT_TYPE,func_arg)[0]) for func_arg in self.function_arg)
    
    def compute_reward(self, action):
        pass
    def is_empty(self):
        return self.function_type==FUNCTION_TYPE["Empty"]
    
def is_boolean_subtask(subtask):
    if subtask.function_type in [FUNCTION_TYPE["Tautology"],FUNCTION_TYPE["CheckDemand"],FUNCTION_TYPE["IsOnFire"]]:
        return True
    else:
        return False
    
class OvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""

    def __init__(self, arglist,subtask_supervised=False,use_map_observation=True):
        self.arglist = arglist
        self.subtask_supervised=subtask_supervised
        self.t = 0

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False
        self.valid_recipes=[]
        self.random_action_prob=arglist.random_action_prob
        
    
        # sub_tasks
        self.all_sub_tasks=[subtask(fun_type,fun_arg) for fun_type,fun_arg in legal_subtasks]
        self.sub_tasks=[subtask(1,1),subtask(2,1),subtask(5,3)]  # You can customize this.
        
        self.use_map_observation=use_map_observation
        
        self.timeout_penalty=0.2
        
        self.env_cache={}

    def __repr__(self):
        return self.world.get_repr() + tuple([agent.get_repr() for agent in self.sim_agents])
    
    def get_repr(self):
        if not self.use_map_observation:
            return self.world.get_repr() + tuple([agent.get_repr() for agent in self.sim_agents])
        else:
            raise NotImplementedError

    def __str__(self):
        # Print the world and agents.
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __eq__(self, other):
        return self.get_repr() == other.get_repr()

    def __copy__(self):
        new_env = OvercookedEnvironment(self.arglist)
        new_env.__dict__ = self.__dict__.copy()
        new_env.world = copy.copy(self.world)
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]

        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                        location=a.location,
                        desired_obj=None,
                        find_held_objects=True)
        return new_env


    def _get_obj_from_str(self,rep:str,location):
        if rep.startswith("F"):
            rep=rep[1:]
        if len(rep)==1 and rep in 'tlopqveh':
            obj=Object(location=location,contents=RepToClass[rep]())
        elif rep=="Cl" or rep=="Ct" or rep=="Co":
            obj = Object(location=location,contents=RepToClass[rep[1]]())
            obj.chop2pieces()
        elif rep=="pCl" or rep=="pCt" or rep=="pCo":
            obj = Object(
                    location=location,
                    contents=RepToClass[rep[2]]())
            obj.chop2pieces()
            obj2 = Object(
                    location=location,
                    contents=RepToClass["p"]())
            obj.merge(obj2)
        elif rep=="CoCt":
            obj = Object(
                    location=location,
                    contents=RepToClass["o"]())
            obj.chop2pieces()
            obj2 = Object(
                    location=location,
                    contents=RepToClass["t"]())
            obj2.chop2pieces()
            obj.merge(obj2)
        elif rep=="pClCt" or rep=="pClCo" or rep=="pCoCt":
            obj = Object(
                    location=location,
                    contents=RepToClass[rep[2]]())
            obj.chop2pieces()
            obj2 = Object(
                    location=location,
                    contents=RepToClass[rep[4]]())
            obj2.chop2pieces()
            obj.merge(obj2)
            obj3=Object(
                    location=location,
                    contents=RepToClass["p"]())
            obj.merge(obj3)
        else:
            print("Unkonwn object:",rep)
            raise NotImplementedError
        return obj
            
    def load_level(self, level, num_agents):
        if self.random_id>=0 and self.random_id in self.env_cache.keys():
            self.world,self.sim_agents=copy.deepcopy(self.env_cache[self.random_id])
            for agent in self.sim_agents:
                if agent.holding is not None:
                    agent.holding = self.world.get_object_at(
                            location=agent.location,
                            desired_obj=None,
                            find_held_objects=True)
            return
        
        x = 0
        y = 0
        if self.random_id>=0:
            map_file=os.path.join('levels/',level,'random{}.txt'.format(self.random_id))
        else:
            raise ValueError("random_id should be larger than 0")
        with open(map_file, 'r') as file:
            # Mark the phases of reading.
            phase = 1
            for line in file:
                line = line.strip('\n')
                if line == '':
                    phase += 1

                # Phase 1: Read in kitchen map.
                elif phase == 1:
                    cur_line_map = []
                    for x, rep in enumerate(line.split(' ')):
                        # Object, i.e. Tomato, Lettuce, Onion, or Plate.
                        if len(rep)==1 and rep in 'tlopqve':
                            counter = Counter(location=(x, y))
                            obj = self._get_obj_from_str(rep, location=(x, y))
                            counter.acquire(obj=obj)
                            self.world.insert(obj=counter)
                            self.world.insert(obj=obj)
                            cur_line_map.append(0)
                        elif len(rep)==1 and rep=='h':
                            f = Floor(location=(x, y))
                            self.world.objects.setdefault('Floor', []).append(f)
                            obj = self._get_obj_from_str(rep, location=(x, y))
                            self.world.insert(obj=obj)
                            f.acquire(obj=obj)
                            cur_line_map.append(1)
                            self.num_fires+=1
                        # GridSquare, i.e. Floor, Counter, Cutboard, Delivery, Pan.
                        elif len(rep)==1 and rep in RepToClass and rep!='f':
                            newobj = RepToClass[rep]((x, y))
                            self.world.objects.setdefault(newobj.name, []).append(newobj)
                            if rep==" ":
                                cur_line_map.append(1)
                            else:
                                cur_line_map.append(0)
                        elif rep=="f":
                            # Empty. Set a Floor tile.
                            f = Floor(location=(x, y))
                            self.world.objects.setdefault('Floor', []).append(f)
                            cur_line_map.append(1)
                        elif rep in ["Cl","Ct","Co","pCl","pCt","pCo","pClCt","pClCo","pCoCt","CoCt","ClCt","ClCo"]:
                            counter = Counter(location=(x, y))
                            obj = self._get_obj_from_str(rep, location=(x, y))
                            counter.acquire(obj=obj)
                            self.world.insert(obj=counter)
                            self.world.insert(obj=obj)
                            cur_line_map.append(0)
                        else:
                            print("Invalid map representation: {}".format(rep))
                    y += 1

                # Phase 2: Read in agent locations (up to num_agents).
                elif phase == 2:
                    if len(self.sim_agents) < num_agents:
                        agent_info = line.split(' ')
                        location=(int(agent_info[0]), int(agent_info[1]))
                        sim_agent = SimAgent(
                                name='agent-'+str(len(self.sim_agents)+1),
                                id_color=COLORS[len(self.sim_agents)],
                                location=location)
                        if len(agent_info)>2:
                            obj = self._get_obj_from_str(agent_info[2], location=location)
                            self.world.insert(obj)
                            sim_agent.acquire(obj)
                            pass
                        self.sim_agents.append(sim_agent)
        self.world.width = x+1
        self.world.height = y
        self.world.perimeter = 2*(self.world.width + self.world.height)
        
        if self.random_id>=0 and self.random_id not in self.env_cache.keys():
            self.env_cache[self.random_id]=(copy.deepcopy(self.world),copy.deepcopy(self.sim_agents))
        return

    def add_random_recipe(self):
        ii=random.randint(0,2)
        self.recipes[ii]+=1
        self.recipes+=1
        
        
    def add_random_fire(self):
        while True:
            xx=random.randint(0,self.world.width-1)
            yy=random.randint(0,self.world.height-1)
            floor = self.world.get_gridsquare_at((xx, yy))
            if isinstance(floor, Floor) and floor.holding==None:
                for agent in self.sim_agents:
                    if agent.location[0]==xx and agent.location[1]==yy:
                        continue
                fire = Object(location=(xx, yy),contents=RepToClass['h']())
                self.world.insert(fire)
                floor.acquire(fire)
                break
        self.num_fires+=1
        return
        
    def reset(self):
        self.world = World(arglist=self.arglist)
        
        self.sim_agents = []
        self.agent_actions = {}
        self.t = 0

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False
        self.last_reward=[0 for i in range(self.num_agents)]
        
        # Load world & distances.
        self.load_level(
                level=self.level,
                num_agents=self.num_agents)
        self.world.make_loc_to_gridsquare()
        return

    def close(self):
        super().close()

        
    def step(self, action_dict):
        raise NotImplementedError
    
    def is_action_possible(self,action_j,agent_i):
        action=ACTION_LIST[action_j%4]
        action_type=ACTION_TYPE[action_j//4]
        agent=self.sim_agents[agent_i]
        world=self.world
        action_x, action_y = world.inbounds(tuple(np.asarray(agent.location) + np.asarray(action)))
        gs = world.get_gridsquare_at((action_x, action_y))
        
        if isinstance(gs, Floor) and action_type=="move" and gs.holding is None: 
            return True
        elif isinstance(gs, Floor) and action_type=="interact" and str(gs.holding)=="h" and str(agent.holding)=="e":
            return True
        elif agent.holding is not None:
            if isinstance(gs, Delivery) and action_type=="serve":
                return agent.holding.is_deliverable()

            elif world.is_occupied(gs.location) and action_type=="merge":
                obj = world.get_object_at(gs.location, None, find_held_objects = False)
                return mergeable(agent.holding, obj)

            elif not world.is_occupied(gs.location):
                obj = agent.holding
                if isinstance(gs, Cutboard) and obj.needs_chopped()  and action_type=="interact":
                    return True
                elif isinstance(gs, Stove) and obj.needs_cooked()  and action_type=="interact":
                    return True
                elif isinstance(gs,Dishwasher) and str(obj)=='q' and action_type=="interact":
                    return True
                elif isinstance(gs,Counter) and action_type=="drop":
                    return True
                elif isinstance(gs,RubbishBin) and str(obj)!='q' and str(obj)!='p' and action_type=="drop":
                    return True
                
        elif agent.holding is None:
            if (isinstance(gs,TomatoSupply) or isinstance(gs,LettuceSupply) or isinstance(gs,OnionSupply)) and action_type=="pick":
                return True
            
            elif world.is_occupied(gs.location) and not isinstance(gs, Delivery) and action_type=="pick" and not str(gs.holding)in ["v","h"]:
                return True
        
        return False
    
    
    def get_action_mask(self):
        action_mask=np.zeros((self.num_agents,24))
        for i in range(self.num_agents):
            for action_j in range(24):
                action_mask[i][action_j]=self.is_action_possible(action_j,i)
            if np.sum(action_mask[i])==0:
                print("stuck in fire")
                action_mask[i][0]=1
        return action_mask
        
    def get_random_action(self):
        action_mask=self.get_action_mask()
        action_dict={}
        for i in range(self.num_agents):
            action_name="agent-{}".format(i+1)
            action=action_mask[i].nonzero()[0][random.randint(0,len(action_mask[i].nonzero()[0])-1)]
            action_dict[action_name]=action
        return action_dict
            
    def interact(self,agent, world):
        """Carries out interaction for this agent taking this action in this world.

        The action that needs to be executed is stored in `agent.action`.
        """
        agent_id=int(agent.name[-1])-1
        some_subroutine_is_completed=False
        sub_rewards={i:0 for i in self.sub_tasks.keys()}
        delivered=False
        # agent does nothing (i.e. no arrow key)
        if agent.action == (0, 0):
            return delivered,sub_rewards,some_subroutine_is_completed
        
        action_x, action_y = world.inbounds(tuple(np.asarray(agent.location) + np.asarray(agent.action)))
        gs = world.get_gridsquare_at((action_x, action_y))
        
        # if floor in front --> move to that square
        if isinstance(gs, Floor) and gs.holding is None:  # not on fire
            agent.move_to(gs.location)
        
        elif isinstance(gs, Floor) and gs.holding is not None and agent.action_type=="interact":  # on fire
            obj=agent.holding
            if str(obj)=='e':
                fire=gs.holding
                gs.release()
                world.remove(fire)
                some_subroutine_is_completed=True
                self.num_fires-=1
                for a_idx,(idx,subtask) in enumerate(self.sub_tasks.items()):
                    if subtask.function_type==FUNCTION_TYPE["PutOffFire"]:
                        sub_rewards[idx]=1
        
        # if holding something
        elif agent.holding is not None:
            # if delivery in front --> deliver
            if isinstance(gs, Delivery) and agent.action_type=="serve":
                obj = agent.holding
                if obj.is_deliverable():
                    some_subroutine_is_completed=True
                    for a_idx,(idx,subtask) in enumerate(self.sub_tasks.items()):
                        if subtask.function_type==FUNCTION_TYPE["Serve"]:
                            if ("p"+get_key(ARGUMENT_TYPE,subtask.function_arg[0])[0]==str(obj)):
                                sub_rewards[idx]=1
                                delivered=True
                    
                    # recipe processing
                    if str(obj)=="pCo":
                        self.recipes[0]=max(0,self.recipes[0]-1)
                    elif str(obj)=="pCt":
                        self.recipes[1]=max(0,self.recipes[1]-1)
                    elif str(obj)=="pCoCt":
                        self.recipes[2]=max(0,self.recipes[2]-1)
                    
                    agent.release()
                    world.remove(obj)
                    obj_q=Object(location=agent.location,
                                contents=RepToClass['q']())
                    world.insert(obj_q)
                    agent.acquire(obj_q)

            # if occupied gridsquare in front --> try merging
            elif world.is_occupied(gs.location) and agent.action_type=="merge":
                # Get object on gridsquare/counter
                obj = world.get_object_at(gs.location, None, find_held_objects = False)
                agent_holder=agent.holding
                obj_cache=copy.copy(obj)
                agent_holder_cache=copy.copy(agent_holder)
                if mergeable(agent.holding, obj):
                    some_subroutine_is_completed=True
                    world.remove(obj)
                    o = gs.release() # agent is holding object
                    world.remove(agent.holding)
                    agent.acquire(obj)
                    world.insert(agent.holding)
                    # if playable version, merge onto counter first
                    for a_idx,(idx,subtask) in enumerate(self.sub_tasks.items()):                      
                        if subtask.function_type==FUNCTION_TYPE["Merge"]:
                            if (merge([get_key(ARGUMENT_TYPE,x)[0] for x in subtask.function_arg]))==merge([str(obj_cache),str(agent_holder_cache)]):
                                sub_rewards[idx]=1
                                break
                            else:
                                sub_rewards[idx]=0


            # if holding something, empty gridsquare in front --> chop or drop
            elif not world.is_occupied(gs.location):
                obj = agent.holding
                if isinstance(gs, Cutboard) and obj.needs_chopped() and agent.action_type=="interact":
                    obj.chop()
                    if obj.is_done():
                        some_subroutine_is_completed=True
                        for a_idx,(idx,subtask) in enumerate(self.sub_tasks.items()):
                            if subtask.function_type==FUNCTION_TYPE["Chop"]:
                                if (get_key(ARGUMENT_TYPE,subtask.function_arg[0])[0].replace("F","C")==str(obj)):
                                    sub_rewards[idx]=1
                                    break

                elif isinstance(gs,Dishwasher) and str(obj)=='q' and agent.action_type=="interact":
                    some_subroutine_is_completed=True
                    agent.release()
                    world.remove(obj)
                    obj = Object(location=agent.location,
                                contents=RepToClass['p']())
                    self.world.insert(obj)
                    agent.acquire(obj)
                    for a_idx,(idx,subtask) in enumerate(self.sub_tasks.items()):
                        if subtask.function_type==FUNCTION_TYPE["Wash"]:
                            sub_rewards[idx]=1
                elif isinstance(gs,Counter) and agent.action_type=="drop":
                    gs.acquire(obj) # obj is put onto gridsquare
                    agent.release()
                    assert world.get_object_at(gs.location, obj, find_held_objects =\
                        False).is_held == False, "Verifying put down works"

        # if not holding anything
        elif agent.holding is None:
            if (isinstance(gs,TomatoSupply) or isinstance(gs,LettuceSupply) or isinstance(gs,OnionSupply)) and agent.action_type=="pick":
                some_subroutine_is_completed=True
                obj=Object(location=agent.location,contents=RepToClass[str(gs).lower()]())
                for a_idx,(idx,subtask) in enumerate(self.sub_tasks.items()):
                    if subtask.function_type==FUNCTION_TYPE["Pick"]:
                        if (get_key(ARGUMENT_TYPE,subtask.function_arg[0])[0].replace("F","")==str(obj)):
                            sub_rewards[idx]=1
                            break
                        else:
                            sub_rewards[idx]=0
                world.insert(obj)
                agent.acquire(obj)
            
            
            # not empty in front --> pick up
            elif world.is_occupied(gs.location) and not isinstance(gs, Delivery) and agent.action_type=="pick":
                obj = world.get_object_at(gs.location, None, find_held_objects = False)
                if not str(obj)in ["v","h"]:
                    for idx,(_,subtask) in enumerate(self.sub_tasks.items()):
                        if subtask.function_type==FUNCTION_TYPE["Pick"]:
                            if (get_key(ARGUMENT_TYPE,subtask.function_arg[0])[0].replace("F","")==str(obj)):
                                sub_rewards[idx]=1
                                break
                            else:
                                sub_rewards[idx]=0
                    gs.release()
                    agent.acquire(obj)

            # if empty in front --> interact
            elif not world.is_occupied(gs.location):
                pass
        
        return delivered,sub_rewards,some_subroutine_is_completed

    def done(self):
        # Done if the episode maxes out
        if self.t >= self.max_timesteps and self.max_timesteps:
            self.termination_info = "Terminating because passed {} timesteps".format(
                    self.max_timesteps)
            self.successful = False
            
        return False

    def reward(self):
        return 1 if self.successful else 0

    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        self.update_display()

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            self.rep[y][x] = str(agent)

    def get_agent_names(self):
        return [agent.name for agent in self.sim_agents]

    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """Returns whether agents are colliding.

        Collisions happens if agent collide amongst themselves or with world objects."""
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # Collision between agents and world objects.
        agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.
            agent1_next_loc = agent1_loc

        agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            agent2_next_loc = agent2_loc

        # Inter-agent collision.
        if agent1_next_loc == agent2_next_loc:
            if agent1_next_loc == agent1_loc and agent1_action != (0, 0):
                execute[1] = False
            elif agent2_next_loc == agent2_loc and agent2_action != (0, 0):
                execute[0] = False
            else:
                # Allow one agent to execute its action.
                rand_i=random.randint(0,1)
                execute[rand_i] = False

        return execute

    def check_collisions(self):
        """Checks for collisions and corrects agents' executable actions.

        Collisions can either happen amongst agents or between agents and world objects."""
        execute = [True for _ in self.sim_agents]

        # Check each pairwise collision between agents.
        for i, j in combinations(range(len(self.sim_agents)), 2):
            agent_i, agent_j = self.sim_agents[i], self.sim_agents[j]
            exec_ = self.is_collision(
                    agent1_loc=agent_i.location,
                    agent2_loc=agent_j.location,
                    agent1_action=agent_i.action,
                    agent2_action=agent_j.action)

            # Update exec array and set path to do nothing.
            if not exec_[0]:
                execute[i] = False
            if not exec_[1]:
                execute[j] = False

            # Track collisions.
            if not all(exec_):
                collision = CollisionRepr(
                        time=self.t,
                        agent_names=[agent_i.name, agent_j.name],
                        agent_locations=[agent_i.location, agent_j.location])
                self.collisions.append(collision)


        # Update agents' actions if collision was detected.
        for i, agent in enumerate(self.sim_agents):
            if not execute[i]:
                agent.action = (0, 0)

    def execute_navigation(self):
        all_delivered=[]
        sub_rewards=[]
        violate_programs=[]
        for agent in self.sim_agents:
            violate_program=False
            delivered,rewards,some_routine_is_completed=self.interact(agent=agent, world=self.world)
            self.agent_actions[agent.name] = agent.action
            all_delivered.append(delivered)
            sub_rewards.append(rewards)
            if some_routine_is_completed and max(rewards.values())<=EPS:
                violate_program=True
            violate_programs.append(violate_program)
        if True in all_delivered:
            delivered=True
        else:
            delivered=False
        return delivered,sub_rewards,violate_programs

def current_subtask_encode(subtasks):
    max_subtask_len=8
    x=np.zeros([max_subtask_len,20])
    cnt=0
    for i,(idx,subtask) in enumerate(subtasks.items()):
        if i>=max_subtask_len:
            break
        x[i][subtask.function_type]=1
        for arg in subtask.function_arg:
            x[i][10+arg]+=1
        cnt+=1
    return x



class ProgramEnvironment(OvercookedEnvironment, ProgramExecutor):
    def __init__(self, cfg,arglist,subtask_supervised,exit_program_penalty,use_program,num_agents=1):
        self.cfg=cfg
        self.partial_obs=cfg.partial_obs
        self.random_map=cfg.random_map
        if self.partial_obs:
            self.map_cc=  np.array([[0,0,0,0,0,0,0,0],
                                    [0,1,1,0,2,2,2,0],
                                    [0,1,1,0,2,2,2,0],
                                    [0,1,1,0,2,2,2,0],
                                    [0,1,1,0,0,2,2,0],
                                    [0,1,1,1,0,2,2,0],
                                    [0,1,1,1,0,2,2,0],
                                    [0,0,0,0,0,0,0,0]])
            self.obs_mask1=   np.array([[1,1,1,1,0,0,0,0],
                                        [1,1,1,1,0,0,0,0],
                                        [1,1,1,1,0,0,0,0],
                                        [1,1,1,1,0,0,0,0],
                                        [1,1,1,1,1,0,0,0],
                                        [1,1,1,1,1,0,0,0],
                                        [1,1,1,1,1,0,0,0],
                                        [1,1,1,1,1,0,0,0]])
            self.obs_mask2=   np.array([[0,0,0,1,1,1,1,1],
                                        [0,0,0,1,1,1,1,1],
                                        [0,0,0,1,1,1,1,1],
                                        [0,0,0,1,1,1,1,1],
                                        [0,0,0,1,1,1,1,1],
                                        [0,0,0,0,1,1,1,1],
                                        [0,0,0,0,1,1,1,1],
                                        [0,0,0,0,1,1,1,1]])
        self.training_stage=cfg.training_stage
        if self.training_stage=="perception":
            self.train_test_split=10000
        else:
            self.train_test_split=18000
        self.arglist=arglist
        self.max_timesteps=cfg.max_timesteps
        self.stage=0
        self.average_score=[1 for _ in range(self.train_test_split)]
        self.level=cfg.level
        self.num_objects=cfg.num_objects
        self.width=cfg.width
        self.height=cfg.height
        self.action_dim=24
        self.subtask_dim=cfg.subtask_dim
        self.device="cuda" if cfg.use_gpu else "cpu"
        if self.random_map:
            max_id=int((self.stage+1)*2000*self.train_test_split/10000)
            probs_sum=sum(self.average_score[:max_id])
            self.random_id=np.random.choice(max_id,1,p=[self.average_score[i]/probs_sum for i in range(max_id)])[0]
        else:
            self.random_id=-1
        if self.training_stage=="evaluate":
            self.random_id=0
        OvercookedEnvironment.__init__(self,arglist,subtask_supervised)
        if self.random_id==-1:
            raise ValueError("random_id should not be -1")
        else:
            ProgramExecutor.__init__(self,program_file=os.path.join('levels/',self.level,'random{}_program.txt'.format(self.random_id)))
        self.sub_tasks=OrderedDict()
        self.exit_program_penalty=exit_program_penalty
        self.use_program=use_program
        self.follow_program_cnt=0
        self.violate_program_cnt=1
        self.last_actions=[4 for i in range(10)]
        self.num_agents=num_agents
        self.last_action_dict=None
        self.force_follow_program=self.arglist.force_follow_program
        self.program_cache={}
        self.program_text_cache={}
        self.first_launch=True
        self.annotations={}
        self.recipes=np.zeros([3,])
        self.num_fires=0
        
        
    def set_testing_samples(self,samples):
        self.test_samples=samples
        self.test_id=0
        
    # curriculum learning, goto next training stage
    def goto_next_stage(self):
        if self.stage<4:
            self.stage+=1
        
    def execute_subroutine(self,subroutine,idx=None):
        if idx is None:
            idx=0
        cur_subtask=subtask(FUNCTION_TYPE[subroutine.ident], [ARGUMENT_TYPE[arg] for arg in subroutine.args],subroutine.lineno)
        self.sub_tasks[idx]=cur_subtask
        return 
        
    def render(self,data=None): 
        if not hasattr(self,"game_screen"):
            self.game_screen=GameScreen(self.world.width,self.world.height)               
        self.game_screen.render(self.world,self.sim_agents,self.t,self.last_reward,self.program_text,self.get_subtask_linenos(),self.last_action_dict,0,self.annotations,self.recipes,self.random_id)
        self.annotations={}
        return self.game_screen.get_img().swapaxes(0,1) 
    
    def render_image_obs(self,data=None): 
        if not hasattr(self,"game_screen"):
            self.game_screen=GameScreen(self.world.width,self.world.height)               
        self.game_screen.render(self.world,self.sim_agents,self.t,self.last_reward,self.program_text,self.get_subtask_linenos(),self.last_action_dict,0,self.annotations,self.recipes,self.random_id)
        return self.game_screen.get_img().swapaxes(0,1)
    
    def execute_till_next_subroutine(self):
        loop_cnt=0
        while False in self.is_stuck.values() or self.is_stuck=={}:
            loop_cnt+=1
            if loop_cnt>10:
                break
            self.execute_one_step()
            if self.exit_program:
                break
        self.sub_tasks=OrderedDict({key:value for key,value in self.sub_tasks.items() if not value.is_empty()})    
    
    def get_fixed_map_observation(self):
        self.map_observation=np.zeros([21,self.world.width,self.world.height])
        for obj in self.world.get_object_list():
            location=obj.location
            obj_abbr=str(obj)
            if obj_abbr in ["f","/","*","w","T","O","v"]:
                if obj_abbr=="f":
                    self.map_observation[0][location[1]][location[0]]=1
                else:
                    self.map_observation[OBJ_TYPE[obj_abbr]][location[1]][location[0]]=1
                
        return self.map_observation
    
    # onehot for merged objects
    def get_repr(self):
        for i in range(21):
            if i not in [0,13,14,15,16,17,18]:   # non-stationary objects
                self.map_observation[i]=0
                
        for obj in self.world.get_object_list():
            obj_abbr=str(obj)
            location=obj.location
            if obj_abbr in ["f","-","/","*","w","T","O","L","v"]:
                continue
            type_id=OBJ_TYPE[obj_abbr]
        
            location=obj.location
            
            self.map_observation[type_id][location[1]][location[0]]=1   # obj_state
            
        obs=dict()
        obs["map"]=self.map_observation[np.newaxis,...].repeat(self.num_agents,axis=0)
        for idx,agent_i in enumerate(self.sim_agents):
            if agent_i.name=="agent-{}".format(idx+1):
                type_id=19
            else:
                type_id=20
            location=agent_i.location
            obs["map"][idx][type_id][location[1]][location[0]]=1
        obs["subtask"]=current_subtask_encode(self.sub_tasks)[:self.num_agents,:]
        obs["recipes"]=self.recipes[np.newaxis,...].repeat(self.num_agents,axis=0)
        obs["action_mask"]=self.get_action_mask()
        obs["agent"]=np.zeros((self.num_agents,3))
        for idx,agent in enumerate(self.sim_agents):
            obs["agent"][idx][0]=agent.location[0]
            obs["agent"][idx][1]=agent.location[1]
            obs["agent"][idx][2]=float(agent.holding is not None)
        obs["timestep"]=np.ones((self.num_agents,1))*self.t
        
        if self.partial_obs:
            location1=self.sim_agents[0].location
            label1=self.map_cc[location1[1]][location1[0]]
            if label1==1:
                obs["map"][0]=obs["map"][0]*self.obs_mask1[np.newaxis,...]
                obs["map"][1]=obs["map"][1]*self.obs_mask2[np.newaxis,...]
            elif label1==2:
                obs["map"][0]=obs["map"][0]*self.obs_mask2[np.newaxis,...]
                obs["map"][1]=obs["map"][1]*self.obs_mask1[np.newaxis,...]
            else:
                raise NotImplementedError
        obs["pss"]=current_subtask_encode(self.sub_tasks)        
        obs["subtask_aux"]=np.zeros((self.num_agents,1))
        if self.num_agents>=2:
            if np.sum(obs["subtask"][1])==0:
                obs["subtask_aux"][1]=1
                obs["subtask"][1]=obs["subtask"][0]
        return obs
    
    def get_obs_keys(self):
        ret = ["map","subtask","recipes","action_mask","agent"]
        return ret
    
    def reset(self,eval=False,derandom=False):
        if self.random_map:
            if eval:
                max_id=20000-1
                self.random_id=random.randint(self.train_test_split+1,max_id)
            else:
                max_id=int((self.stage+1)*2000*self.train_test_split/10000)
                probs_sum=sum(self.average_score[:max_id])
                self.random_id=np.random.choice(max_id,1,p=[self.average_score[i]/probs_sum for i in range(max_id)])[0]
            if derandom:
                self.random_id=self.test_samples[self.test_id]
                self.test_id+=1
        else:
            self.random_id=-1
        self.num_fires=0
        OvercookedEnvironment.reset(self)
        if self.random_id==-1:
            raise ValueError("random_id should not be -1")
        else:
            if self.random_id in self.program_cache.keys():
                ProgramExecutor.__init__(self,program_file=os.path.join('levels/',self.level,'random{}_program.txt'.format(self.random_id)),cached_program=self.program_cache[self.random_id],cached_program_text=self.program_text_cache[self.random_id])
            else:
                ProgramExecutor.__init__(self,program_file=os.path.join('levels/',self.level,'random{}_program.txt'.format(self.random_id)))
                self.program_cache[self.random_id]=copy.deepcopy(self.program)
                self.program_text_cache[self.random_id]=copy.deepcopy(self.program_text)
        self.exit_program=False
        self.sub_tasks=OrderedDict()
        self.recipes=np.zeros([3,])
        for _ in range(1):
            self.add_random_recipe()
        self.execute_till_next_subroutine()
        self.last_actions=[0 for i in range(10)]
        self.last_action_dict=None
        self.get_fixed_map_observation()
        self.first_launch=False
        obs=self.get_repr()
        return obs


    # random allocate tasks to agents
    def random_allocate(self,tasks):
        tasks_all=tasks[0].copy()
        num_tasks=np.sum(np.sum(tasks_all,axis=1)>0)
        for i in range(self.num_agents):
            tasks[i].fill(0)
            tasks[i][0]=tasks_all[i]
            if np.sum(tasks_all[i])==0:
                tasks[i][0][0]=1
                tasks[i][0][10]=1
            
        return tasks
        
    def get_subtask_linenos(self):
        lineno=[]
        for key in self.sub_tasks:
            lineno.append(self.sub_tasks[key].lineno)
        return list(set(lineno))
        
        
    def is_branching(self):
        for key in self.sub_tasks:
            if is_boolean_subtask(self.sub_tasks[key]):
                return True
        return False
    
    
    def compute_perception_gt(self):
        answer=False
        for idx,(key,value) in enumerate(self.sub_tasks.items()):
            assert is_boolean_subtask(value)
            if value.function_type==FUNCTION_TYPE["Tautology"]:
                answer=True
            elif value.function_type==FUNCTION_TYPE["CheckDemand"]:
                if value.function_arg[0]==ARGUMENT_TYPE["pCo"] and self.recipes[0]>0:
                    answer=True
                if value.function_arg[0]==ARGUMENT_TYPE["pCt"] and self.recipes[1]>0:
                    answer=True
                if value.function_arg[0]==ARGUMENT_TYPE["pCoCt"] and self.recipes[2]>0:
                    answer=True
            elif value.function_type==FUNCTION_TYPE["IsOnFire"] and self.num_fires>0:
                answer=True
        self.exit_program=True
        return answer
                
    def step_perception(self, perception):
        perception=torch.round(perception)
        cnt=0
        to_delete_keys=[]
        for idx,(key,value) in enumerate(self.sub_tasks.items()):
            if is_boolean_subtask(value):
                if  value.function_type==FUNCTION_TYPE["Tautology"]:
                    perception[cnt]=1
                if value.function_type==FUNCTION_TYPE["CheckDemand"]:
                    if value.function_arg[0]==ARGUMENT_TYPE["pCo"] and self.recipes[0]>0:
                        perception[cnt]=1
                    elif value.function_arg[0]==ARGUMENT_TYPE["pCt"] and self.recipes[1]>0:
                        perception[cnt]=1
                    elif value.function_arg[0]==ARGUMENT_TYPE["pCoCt"] and self.recipes[2]>0:
                        perception[cnt]=1
                    else:
                        perception[cnt]=0
                if value.function_type==FUNCTION_TYPE["IsOnFire"]:
                    if self.num_fires>0:
                        perception[cnt]=1
                    else:
                        perception[cnt]=0
                    
                if perception[cnt]>0.9:
                    self.cond[key]=True
                else:
                    self.cond[key]=False
                self.is_stuck[key]=False
                cnt+=1
                to_delete_keys.append(key)
        for key in to_delete_keys:
            self.sub_tasks.pop(key)
        self.execute_till_next_subroutine()
        return self.exit_program
        
        
    def random_step(self):
        random_action=self.get_random_action()
        self.step(random_action)
        return self.get_repr()
        
    def get_obs_shape(self):
        ret=dict()
        ret["map"]=(self.num_objects,self.width,self.height)
        ret["recipes"]=(3,)
        ret["subtask"]=(self.subtask_dim,)
        ret["subtask_aux"]=(1,)
        ret["action_mask"]=(self.action_dim,)
        ret["agent"]=(3,)
        ret["timestep"]=(1,)
        return ret
        
    def step(self, action_dict):
        self.last_action_dict=action_dict

        # Get actions.
        for sim_agent in self.sim_agents:
            sim_agent.action = ACTION_LIST[action_dict[sim_agent.name]%4]
            sim_agent.action_type=ACTION_TYPE[action_dict[sim_agent.name]//4]
            if random.random()<self.random_action_prob:
                sim_agent.action = random.choice(ACTION_LIST)
                sim_agent.action_type=random.choice(ACTION_TYPE)

            
        # Check collisions.
        self.check_collisions()

        # Execute.
        _,sub_rewards,violate_programs=self.execute_navigation()
        
            
        if not self.use_program:
            sub_rewards=[{j:0 for j in self.sub_tasks.keys()} for i in range(len(self.sim_agents) )]
        
        # First, judge whether any subtask is violated,including the subtask being completed by the wrong agent
        is_violated=False
        if self.force_follow_program:
            for i in range(self.num_agents):
                if(violate_programs[i]):
                    is_violated=True
                    self.exit_program=True
                    done=True
                    if done and self.train_test_split>self.random_id>=0:
                        self.average_score[self.random_id]=0.9999*self.average_score[self.random_id]+0.0001*(1-0.995**512)

        # Second, forward the program and see whether the task is completed.
        if not is_violated:       
            stuck=True
            done=False
            for ii,_ in self.sub_tasks.items():
                for jj in range(len(self.sim_agents)):
                    if sub_rewards[jj][ii]==1:
                        self.sub_tasks[ii]=subtask(0,[0,])
                        self.is_stuck[ii]=False
                        stuck=False
            if not stuck:
                self.execute_till_next_subroutine()      
                done=True 
                
            # curriculum learning; used in training
            if done and self.train_test_split>self.random_id>=0:
                self.average_score[self.random_id]=0.9*self.average_score[self.random_id]+0.1*(1-0.995**self.t)
                
                
        # For both violating or obeying the program, the env is terminated.
        is_terminal=done      
        
        # Third, compute the reward
        if self.subtask_supervised:
            reward=[sum(list(sub_rewards[i].values()))*1 for i in range(len(self.sim_agents))]
            if sum(reward)>0:
                total_reward=sum(reward)
                reward=[reward[i]+(total_reward-reward[i])*0.2 for i in range(len(self.sim_agents))]
        else:
            reward=[0 for i in range(len(self.sim_agents))]
            
            
        # used in training 
        for i in range(self.num_agents):
            if(violate_programs[i]):
                reward[i]-=0.05
        
        
        # used in training; encourage exploration
        self.t+=1                  
        for i in range(self.num_agents):
            reward[i]-=0.01

        self.last_reward=reward
            
            
        
        self.last_actions.pop(0)
        self.last_actions.append(action_dict['agent-1'])
        obs=self.get_repr()
        if done and not is_violated:
            obs["subtask"].fill(0)
            obs["subtask"][:,0]=1
            obs["subtask"][:,10]=1
            
            
        if self.t >= self.max_timesteps and self.max_timesteps and not done:
            done=True
            is_terminal=False
            self.exit_program=True
          
          
        # Testing Fire Accident
        done=self.exit_program  
        if self.num_fires==0 and self.t==60:
            self.add_random_fire()
            
        return   obs, np.stack(reward), np.stack([done for i in range(len(self.sim_agents))]), np.stack([is_terminal for i in range(len(self.sim_agents))]),None