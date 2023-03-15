import os
import numpy as np
from typing import List, Dict, Tuple
import random
from math import *
import gym
import argparse
import time
import numpy as np
import random
import argparse
from collections import namedtuple
import gym
import os
import tqdm

def soft_binarization(x,tau=3):
    return exp(tau*x)/(exp(tau*x)+exp(tau*(1-x)))


initial_holding_allowed=False
    
class Map():
    def __init__(self,map_size:Tuple[int,int]):
        self.map_size=map_size
        self.map=[['-' for i in range(map_size[0])] for j in range(map_size[1])]
        self.agent=[]
        self.num_agents=2
        
    def is_boundary(self,i,j):
        return i==0 or j==0 or i==self.map_size[0]-1 or j==self.map_size[1]-1
    
    def build_boundary(self):
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.is_boundary(i,j):
                    self.map[i][j]='-'
                
    
    # p: percentage of obstacles
    def generate_path(self,p=0.3):
        dist=1/3*(self.map_size[0]+self.map_size[1])
        self.points=[]
        # modify=log(log(self.map_size[0]*self.map_size[1]))
        modify=1
        cnt=round(self.map_size[0]*self.map_size[1]*(1-p)/dist*modify)
        cnt=max(cnt,2)
        for i in range(cnt):
            point=(1+round((self.map_size[0]-3)*soft_binarization(random.random())),1+round((self.map_size[1]-3)*soft_binarization(random.random())))
            self.points.append(point)
            self.map[point[0]][point[1]]='f'
        self.points.append(self.points[0])
        for i in range(cnt):
            cur_x=self.points[i][0]
            cur_y=self.points[i][1]
            while cur_x!=self.points[i+1][0] or cur_y!=self.points[i+1][1]:
                delta_x=np.sign(self.points[i+1][0]-cur_x)
                delta_y=np.sign(self.points[i+1][1]-cur_y)
                if random.random()<0.5:
                    cur_x=cur_x+delta_x
                else:
                    cur_y=cur_y+delta_y
                self.map[cur_x][cur_y]='f'
                
    def check_connectivity(self,p):
        vis=np.zeros([self.map_size[0],self.map_size[1]],np.int32)
        cc=0
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if p[i][j]>0 and vis[i][j]==0:
                    cc+=1
                    dfs_list=[(i,j)]
                    while True:
                        if len(dfs_list)==0:
                            break
                        for ii,jj in dfs_list:
                            if vis[ii][jj]!=0:
                                dfs_list.remove((ii,jj))
                                continue
                            vis[ii][jj]=cc
                            for iii,jjj in [(ii+1,jj),(ii,jj+1),(ii-1,jj),(ii,jj-1)]:
                                if p[iii][jjj]==1 and vis[iii][jjj]==0:
                                    dfs_list.append((iii,jjj))
        for i in range(1,cc+1):
            if np.sum(vis==i)<=2:
                return 0   # CC that consists of less than 2 grids is not allowed
        self.vis=vis
        return cc
                    
                    
        
    def generate_convex_area(self):
        random_img=np.zeros([self.map_size[0],self.map_size[1]],np.int32)
        num_pts=random.randint(6,10)
        for i in range(num_pts):
            random_img[random.randint(1,self.map_size[0]-2)][random.randint(1,self.map_size[1]-2)]=1
        
        element_count=np.sum(random_img)
        while True:
            for i in range(self.map_size[0]):
                idx1=np.argwhere(random_img[i]>0)
                if len(idx1)==0:
                    continue
                idx_min=np.min(idx1)
                idx_max=np.max(idx1)
                random_img[i][idx_min:idx_max+1]=1
            for j in range(self.map_size[1]):
                idx1=np.argwhere(random_img[:,j]>0)
                if len(idx1)==0:
                    continue
                idx_min=np.min(idx1)
                idx_max=np.max(idx1)
                random_img[idx_min:idx_max+1][:,j]=1
            if np.sum(random_img)==element_count:
                break
            element_count=np.sum(random_img)
        
        cc=self.check_connectivity(random_img)
        if cc!=1:
            raise Exception("CC={}".format(cc))
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if random_img[i][j]>0:
                    self.map[i][j]='f'
        
    def generate_two_cc(self):
        while True:
            random_img=np.zeros([self.map_size[0],self.map_size[1]],np.int32)
            for i in range(1,self.map_size[0]-1):
                for j in range(1,self.map_size[1]-1):
                    if random.randint(0,1)==0:
                        random_img[i][j]=1
        
            cc=self.check_connectivity(random_img)
            if cc==2:
                break
            # else:
            #     print("CC={}".format(cc))
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if random_img[i][j]>0:
                    self.map[i][j]='f'

             
             
    def generate_fixed_path(self):
        self.map=[['-','-','-','-','-','-','-','-'],
                  ['-','f','f','-','f','f','f','-'],
                  ['-','f','f','-','f','f','f','-'],
                  ['-','f','f','-','f','f','f','-'],
                  ['-','f','f','-','-','f','f','-'],
                  ['-','f','f','f','-','f','f','-'],
                  ['-','f','f','f','-','f','f','-'],
                  ['-','-','-','-','-','-','-','-']]
                    
        self.vis=[[0,0,0,0,0,0,0,0],
                  [0,1,1,0,2,2,2,0],
                  [0,1,1,0,2,2,2,0],
                  [0,1,1,0,2,2,2,0],
                  [0,1,1,0,0,2,2,0],
                  [0,1,1,1,0,2,2,0],
                  [0,1,1,1,0,2,2,0],
                  [0,0,0,0,0,0,0,0]]
        
    def save(self,path:str="./utils/levels/random/example.txt"):
        with open(path,"w") as f:
            f.write('\n'.join([' '.join(x) for x in self.map]))
            f.write('\n\n')
            
            for idx,agent in enumerate(self.agent):
                f.write(str(agent[0])+" "+str(agent[1]))
                if initial_holding_allowed:
                    if random.random()<0.2:
                        f.write(" "+random.choice(["o","t","Co","Ct","p","q","pCo","pCt","CoCt","pCoCt","e"]))
                f.write('\n')
    def cnt_grid(self):
        return self.map_size[0]*self.map_size[1]
    
    def is_occupied(self,i,j):
        return self.map[i][j]!='f' or (j,i) in self.agent
    
    def count_empty(self):
        cnt=0
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if not self.is_occupied(i,j):
                    cnt+=1
        return cnt
    
    def count_empty_i(self,idx):
        cnt=0
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if not self.is_occupied(i,j) and self.vis[i][j]==idx:
                    cnt+=1
        return cnt
    
    def add_agent(self):
        empty_cnt=self.count_empty()
        x=random.randint(0,empty_cnt-1)
        cnt=0
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if not self.is_occupied(i,j):
                    if(cnt==x):
                        self.agent.append((j,i))
                        return
                    cnt+=1
                    
    def add_agent_i(self,idx):
        empty_cnt=self.count_empty_i(idx)
        x=random.randint(0,empty_cnt-1)
        cnt=0
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if not self.is_occupied(i,j) and self.vis[i][j]==idx:
                    if(cnt==x):
                        self.agent.append((j,i))
                        return
                    cnt+=1               
    
    
    def random_accessible_center(self):
        cnt=1
        point=None
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.map[i][j]=="-":
                    if self.map[max(i-1,0)][j]=='f' or self.map[min(i+1,self.map_size[0]-1)][j]=='f' or self.map[i][max(j-1,0)]=='f' or self.map[i][min(j+1,self.map_size[1]-1)]=='f':
                        if random.random()<1/cnt:
                            point=(i,j)
                        cnt+=1
        if point is None:
            raise ValueError
        return point
    
    def random_accessible_bridge(self):
        cnt=1
        point=None
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.map[i][j]=="-":
                    reachable=True
                    for agent_id in range(1,3):
                        if not(self.vis[max(i-1,0)][j]==agent_id or self.vis[min(i+1,self.map_size[0]-1)][j]==agent_id or self.vis[i][max(j-1,0)]==agent_id or self.vis[i][min(j+1,self.map_size[1]-1)]==agent_id):
                            reachable=False
                    if reachable:
                        if random.random()<1/cnt:
                            point=(i,j)
                        cnt+=1
        if point is None:
            raise ValueError
        return point
    
    def random_accessible_i(self,idx):
        if not isinstance(idx,List):
            idx=[idx]
        cnt=1
        point=None
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.map[i][j]=="-":
                    reachable=True
                    for agent_id in idx:
                        if not(self.vis[max(i-1,0)][j]==agent_id or self.vis[min(i+1,self.map_size[0]-1)][j]==agent_id or self.vis[i][max(j-1,0)]==agent_id or self.vis[i][min(j+1,self.map_size[1]-1)]==agent_id):
                            reachable=False
                    if not (self.map[max(i-1,0)][j]=='f' or self.map[min(i+1,self.map_size[0]-1)][j]=='f' or self.map[i][max(j-1,0)]=='f' or self.map[i][min(j+1,self.map_size[1]-1)]=='f'):
                        reachable=False
                    if reachable:
                        if random.random()<1/cnt:
                            point=(i,j)
                        cnt+=1
        if point is None:
            raise ValueError
        return point
    
    def add_delivery(self,idx):
        point=self.random_accessible_i(idx)
        self.map[point[0]][point[1]]='*'
        
    def add_dishwasher(self,idx):
        point=self.random_accessible_i(idx)
        self.map[point[0]][point[1]]='w'
        
    def add_fire(self,idx):
        cnt=1
        point=None
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.map[i][j]=="f" and self.vis[i][j]==idx and not (self.map[max(i-1,0)][j]=='e' or self.map[min(i+1,self.map_size[0]-1)][j]=='e' or self.map[i][max(j-1,0)]=='e' or self.map[i][min(j+1,self.map_size[1]-1)]=='e'):
                    if random.random()<1/cnt:
                        point=(i,j)
                    cnt+=1
        self.map[point[0]][point[1]]='h'
        
    def add_plate(self,idx):
        point=self.random_accessible_i(idx)
        self.map[point[0]][point[1]]='p'
        
    def add_random_plate(self,idx):
        point=self.random_accessible_i(idx)
        if random.randint(0,1)==1:
            self.map[point[0]][point[1]]='p'
        else:
            self.map[point[0]][point[1]]='q'
            
    def add_random_dish(self,idx):
        point=self.random_accessible_i(idx)
        x=random.randint(0,3)
        if x==0:
            self.map[point[0]][point[1]]='pCt'
        elif x==1:
            self.map[point[0]][point[1]]='pCo'
        elif x==2:
            self.map[point[0]][point[1]]='pCoCt'
        elif x==3:
            self.map[point[0]][point[1]]='CoCt'
            
    def add_random_chopped(self,idx):
        point=self.random_accessible_i(idx)
        if random.randint(0,1)==1:
            self.map[point[0]][point[1]]='Ct'
        else:
            self.map[point[0]][point[1]]='Co'
        
    def add_random_fresh(self,idx):
        point=self.random_accessible_i(idx)
        if random.randint(0,1)==1:
            self.map[point[0]][point[1]]='t'
        else:
            self.map[point[0]][point[1]]='o'
        
        
    def add_object_str(self,s,idx):
        point=self.random_accessible_i(idx)
        self.map[point[0]][point[1]]=s
        
    def add_vases(self):
        self.map[4][3]='v'
        self.map[4][4]='v'


    

    


if __name__=="__main__":
    # training levels for different training stages
    # sast: single agent single subtask, train single agent policies
    # samt: single agent multi subtask, train single agent policies on completing series of subtasks
    # mast: multi agent single subtask
    # mast-cooperation: multi agent single subtask, agent 1 is the leading agent, agent 2 is the follower agent
    # perception: train the perception module
    # mamt: multi agent multi subtask, integrate all the modules
    level=input("level: sast | samt | mast | mast-cooperation | mamt | perception \n")
    
    os.makedirs(os.path.join("./levels/",level),exist_ok=True)

    if level in ["perception","sast","mast","mast-cooperation"]:
        initial_holding_allowed=True
        
    program_dir="./programs"
    
    for i in tqdm.tqdm(range(0,20000)):
        while True:
            try:
                random_cc_agent=random.randint(1,2)
                if level=="sast" or level=="samt":
                    random_cc_fixed=random_cc_agent
                    random_cc_movable=random_cc_agent
                elif level in ["mast","mast","perception","mamt"]:
                    random_cc_fixed=[]
                    random_cc_movable=[]
                elif level=="mast-cooperation":
                    random_cc_fixed=random_cc_agent
                    random_cc_movable=[]
                else:
                    raise ValueError("level not found")
                name="random"+str(i)
                map=Map((8,8))
                map.generate_fixed_path()
                map.add_vases()
                
                map.add_object_str("*",random_cc_fixed)
                map.add_object_str("w",random_cc_fixed)
                map.add_object_str("T",random_cc_fixed)
                map.add_object_str("O",random_cc_fixed)
                map.add_object_str("/",random_cc_fixed)
                map.add_object_str("e",random_cc_fixed)
                
                
                map.add_agent_i(random_cc_agent)
                map.add_agent_i(3-random_cc_agent)
                map.add_agent_i(random_cc_agent)
                map.add_agent_i(3-random_cc_agent)
                
                # randomly generate one agent and one subtask
                if level in ["sast","mast-cooperation","mast"]:
                    h1=0
                    h2=random.randint(0,15)
                    if h2==0:
                        map.add_object_str("pCoCt",random_cc_movable)
                    if h2==1:
                        map.add_object_str("pCo",random_cc_movable)
                    if h2==2:
                        map.add_object_str("pCt",random_cc_movable)
                    if h2==3:
                        map.add_object_str("p",random_cc_movable)
                        map.add_object_str("CoCt",random_cc_movable)
                    if h2==4:
                        map.add_object_str("p",random_cc_movable)
                        map.add_object_str("Ct",random_cc_movable)
                        map.add_object_str("Co",random_cc_movable)
                    if h2==5:
                        map.add_random_plate(random_cc_movable)
                        map.add_object_str("Co",random_cc_movable)
                        map.add_object_str("Ct",random_cc_movable)
                    if h2==6:
                        map.add_object_str("p",random_cc_movable)
                        map.add_object_str("Co",random_cc_movable)
                        map.add_object_str("Ct",random_cc_movable)
                    if h2==7:
                        map.add_object_str("pCo",random_cc_movable)
                        map.add_object_str("Ct",random_cc_movable)
                    if h2==8:
                        map.add_object_str("pCt",random_cc_movable)
                        map.add_object_str("Co",random_cc_movable)
                    if h2==9 or h2==10:
                        map.add_random_plate(random_cc_movable)
                    if h2==11:
                        map.add_random_plate(random_cc_movable)
                        map.add_object_str("t",random_cc_movable)
                        map.add_object_str("o",random_cc_movable)
                    if h2==12:
                        map.add_random_plate(random_cc_movable)
                        map.add_object_str("t",random_cc_movable)
                        map.add_object_str("o",random_cc_movable)
                    if h2==13 or h2==14:
                        map.add_object_str("q",random_cc_movable)
                        map.add_random_chopped(random_cc_movable)
                    if h2==15:
                        map.add_fire(random_cc_fixed)
                    for _ in range(random.randint(1,3)):
                        map.add_random_dish([])
                    if not level=="sast":
                        if random.random()<0.1 and not h2==15:
                            map.add_fire(random.randint(0,1))
                
                # randomly generate a perception query
                elif level in ["perception"]:
                    h1=5
                    h2=random.randint(1,4)
                    for _ in range(3):
                        map.add_random_dish([])
                    for _ in range(3):
                        map.add_random_chopped([])
                    if random.random()<0.5:
                        map.add_fire(random.randint(0,1))
                
                
                elif level in ["samt","mast-cooperation"]:
                    h1=1
                    h2=random.randint(1,13)
                    if h2==3:
                        map.add_object_str("q",random_cc_movable)
                        map.add_object_str("Ct",random_cc_movable)
                    if h2==4:
                        map.add_object_str("q",random_cc_movable)
                        map.add_object_str("Co",random_cc_movable)
                    if h2==5:
                        map.add_object_str("p",random_cc_movable)
                        map.add_object_str("Co",random_cc_movable)
                    if h2==6:
                        map.add_object_str("p",random_cc_movable)
                        map.add_object_str("Ct",random_cc_movable)
                    if h2==7:
                        map.add_object_str("CoCt",random_cc_movable)
                        map.add_object_str("p",random_cc_movable)
                    if h2==8:
                        map.add_fire(random_cc_movable)
                    if h2==9:
                        map.add_fire(random_cc_movable)
                        map.add_object_str("o",random_cc_movable)
                    if h2==10:
                        map.add_fire(random_cc_movable)
                        map.add_object_str("Ct",random_cc_movable)
                        map.add_object_str("p",random_cc_movable)
                    if h2==11:
                        map.add_fire(random_cc_movable)
                        map.add_object_str("pCoCt",random_cc_movable)
                    if h2==12:
                        map.add_fire(random_cc_movable)
                    if h2==13:
                        map.add_fire(random_cc_movable)
                        map.add_object_str("q",random_cc_movable)
                        
                elif level in ["mamt"]:
                    h1=4
                    h2=random.randint(0,3)
                    map.add_object_str("p",random_cc_movable)
                    map.add_object_str("q",random_cc_movable)
                break
            except:
                continue
        
        
        program=open(os.path.join(program_dir,str(h1)+"_"+str(h2)+".txt")).read()        
        map.save(os.path.join("./levels/",level,name+".txt"))
        with open(os.path.join("./levels/",level,name+"_program"+".txt"),"w") as f:
            f.write(program)
        