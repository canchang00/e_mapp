import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from utils.core import *
from misc.game.utils import *
import math

graphics_dir = 'misc/game/graphics'
_image_library = {}

ACTION_LIST=[[0,1],[1,0],[0,-1],[-1,0],[0,0]]

def get_image(path):
    global _image_library
    image = _image_library.get(path)
    if image == None:
        canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
        image = pygame.image.load(canonicalized_path)
        _image_library[path] = image
    return image

class Color:
    BLACK = (0, 0, 0)
    FLOOR = (245, 230, 210)  # light gray
    COUNTER = (220, 170, 110)   # tan/gray
    COUNTER_BORDER = (114, 93, 51)  # darker tan
    DELIVERY = (96, 96, 96)  # grey
    

def arrow(screen, lcolor, tricolor, start, end, trirad=10, thickness=1):
    rad=math.pi/180
    pygame.draw.line(screen, lcolor, start, end, thickness)
    rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi/2
    pygame.draw.polygon(screen, tricolor, ((end[0] + trirad * math.sin(rotation),
                                        end[1] + trirad * math.cos(rotation)),
                                    (end[0] + trirad * math.sin(rotation - 120*rad),
                                        end[1] + trirad * math.cos(rotation - 120*rad)),
                                    (end[0] + trirad * math.sin(rotation + 120*rad),
                                        end[1] + trirad * math.cos(rotation + 120*rad))))  
class GameScreen:
    def __init__(self,n_width,n_height):
        # Visual parameters
        self.scale = int(80/n_width*6)   # num pixels per tile
        self.fixed_scale=80
        self.holding_scale = 0.5
        self.container_scale = 0.7
        self.n_width = n_width
        self.n_height= n_height+2
        self.width = self.scale * self.n_width+self.fixed_scale*4
        self.height = self.scale * self.n_height
        self.tile_size = (self.scale, self.scale)
        self.holding_size = tuple((self.holding_scale * np.asarray(self.tile_size)).astype(int))
        self.container_size = tuple((self.container_scale * np.asarray(self.tile_size)).astype(int))
        self.holding_container_size = tuple((self.container_scale * np.asarray(self.holding_size)).astype(int))
        self.screen = pygame.Surface((self.width, self.height))
        pygame.font.init()
        self.font = pygame.font.SysFont('Courier', 15)

    def get_img(self):
        return pygame.surfarray.array3d(self.screen)
    
    def render(self,world,sim_agents,t,last_reward,program_text,linenos,last_action_dict,allocate_id0,annotation,recipes,random_id):
        annotation={}
        self.screen.fill(Color.FLOOR)
        for obj in world.get_object_list():
            obj_abbr=str(obj)
            location=obj.location
            if obj_abbr=="-":
                self.draw_counter(location)
            elif obj_abbr=="/":
                self.draw_counter(location)
                self.draw_cutboard(location)
            elif obj_abbr=="*":
                self.draw_counter(location)
                self.draw_delivery(location)
            elif obj_abbr=="s":
                self.draw_counter(location)
                self.draw_stove(location)
            elif obj_abbr=="w":
                self.draw_counter(location)
                self.draw_dishwasher(location)
            elif obj_abbr=="T":
                self.draw_counter(location)
                self.draw_tomato_supply(location)
            elif obj_abbr=="O":
                self.draw_counter(location)
                self.draw_onion_supply(location)
            elif obj_abbr=="L":
                self.draw_counter(location)
                self.draw_lettuce_supply(location)
            elif obj_abbr=="R":
                self.draw_counter(location)
                self.draw_rubbish_bin(location)
            elif obj_abbr=="v":
                self.draw_counter(location)
                self.draw_vase(location)
            elif obj_abbr=="h":
                self.draw_fire(location)
            elif obj_abbr=="f":
                continue
            elif obj.is_held==False:
                self.draw_object(obj)
        
        for agent in sim_agents:
            self.draw_agent(agent,None if last_action_dict is None else last_action_dict[agent.name])
        
        time_text_display = self.font.render(f"time:{t}",False,(0,0,0,0.5))
        self.screen.blit(time_text_display,[5,5])
        reward_mes="reward: "+"v.s.".join(["{:.2f}".format(r) for r in last_reward])
        reward_text_display = self.font.render(reward_mes,False,(0,0,0,0.5))
        self.screen.blit(reward_text_display,[5,20])
        env_id_text_display = self.font.render(f"env_id:{random_id}",False,(0,0,0,0.5))
        self.screen.blit(env_id_text_display,[5,35])
        for idx,(key,value) in enumerate(annotation.items()):
            mes=str(key)+": "+"-".join(["{:.2f}".format(d) for d in value])
            distance_text_display = self.font.render(mes,False,(0,0,0,0.5))
            self.screen.blit(distance_text_display,[5,50+15*idx])
        recipe_mes="recipe: "+f"O:{recipes[0]} T:{recipes[1]} O+T:{recipes[2]}"
        recipe_text_display=self.font.render(recipe_mes,False,(0,0,0,0.5))
        self.screen.blit(recipe_text_display,[5,self.scale*2-20])
        
        
        program_text_lines=program_text.split('\n')
        line_counts=0
        for idx,line in enumerate(program_text_lines):
            if idx+1 in linenos:
                program_text_display = self.font.render(line,False,(255,0,0,0.2))
                line_counts+=1
            else:
                program_text_display = self.font.render(line,False,(0,0,0,0.2))
            self.screen.blit(program_text_display,[self.scale*self.n_width+self.fixed_scale*0.5,self.fixed_scale*0.5+idx*10])
            
        self.draw_borderline()


    def draw_borderline(self,):
        for i in range(self.n_width+1):
            for j in range(self.n_height+1):
                sl=self.scaled_location((i,j))
                pygame.draw.line(self.screen, (30,30,30,1), (sl[0],sl[1]), (sl[0],sl[1]+self.scale),2)
        for i in range(self.n_width):
            for j in range(self.n_height+1):
                sl=self.scaled_location((i,j))
                pygame.draw.line(self.screen, (30,30,30,1), (sl[0],sl[1]), (sl[0]+self.scale,sl[1]),2)
                
    def draw_counter(self, location):
        sl = self.scaled_location(location)
        fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
        
        pygame.draw.rect(self.screen, Color.COUNTER, fill)
        
    def draw_cutboard(self,location):
        self.draw('cutboard', self.tile_size, self.scaled_location(location))

    def draw_dishwasher(self,location):
        self.draw('dishwasher', self.tile_size, self.scaled_location(location))
        
    def draw_stove(self,location):
        self.draw('stove', self.tile_size, self.scaled_location(location))
        
    def draw_tomato_supply(self,location):
        self.draw('TomatoSupply', self.tile_size, self.scaled_location(location))
        
    def draw_onion_supply(self,location):
        self.draw('OnionSupply', self.tile_size, self.scaled_location(location))
        
    def draw_lettuce_supply(self,location):
        self.draw('LettuceSupply', self.tile_size, self.scaled_location(location))
        
    def draw_delivery(self, location):
        self.draw('delivery', self.tile_size, self.scaled_location(location))        

    def draw_rubbish_bin(self, location):
        self.draw('bin', self.tile_size, self.scaled_location(location))     
        
    def draw_vase(self, location):
        self.draw('vase', self.tile_size, self.scaled_location(location))  
        
    def draw_fire(self, location):
        self.draw('fire', self.tile_size, self.scaled_location(location))  
        
    def draw(self, path, size, location):
        image_path = '{}/{}.png'.format(graphics_dir, path)
        image = pygame.transform.scale(get_image(image_path), size)
        self.screen.blit(image, location)
    
    def save_img(self,path):
        pygame.image.save(self.screen, path)
        
    def draw_agent(self, agent,last_action=None):
        setattr(pygame.draw, 'arrow', arrow)
        self.draw('agent-{}'.format(agent.color),
            self.tile_size, self.scaled_location(agent.location))
        self.draw_agent_object(agent.holding)
            
    
    def draw_agent_object(self, obj):
        # Holding shows up in bottom right corner.
        if obj is None: return
        if any([isinstance(c, Plate) for c in obj.contents]): 
            self.draw('Plate', self.holding_size, self.holding_location(obj.location))
            if len(obj.contents) > 1:
                plate = obj.unmerge('Plate')
                self.draw(obj.full_name, self.holding_container_size, self.holding_container_location(obj.location))
                obj.merge(plate)
        else:
            self.draw(obj.full_name, self.holding_size, self.holding_location(obj.location))

    def draw_object(self, obj):
        if obj is None: return
        if any([isinstance(c, Plate) for c in obj.contents]): 
            self.draw('Plate', self.tile_size, self.scaled_location(obj.location))
            if len(obj.contents) > 1:
                plate = obj.unmerge('Plate')
                self.draw(obj.full_name, self.container_size, self.container_location(obj.location))
                obj.merge(plate)
        else:
            self.draw(obj.full_name, self.tile_size, self.scaled_location(obj.location))

    def scaled_location(self, loc):
        """Return top-left corner of scaled location given coordinates loc, e.g. (3, 4)"""
        loc=(loc[0],loc[1]+2)
        return tuple(self.scale * np.asarray(loc))

    def holding_location(self, loc):
        """Return top-left corner of location where agent holding will be drawn (bottom right corner) given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.scale*(1-self.holding_scale)).astype(int))

    def container_location(self, loc):
        """Return top-left corner of location where contained (i.e. plated) object will be drawn, given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.scale*(1-self.container_scale)/2).astype(int))

    def holding_container_location(self, loc):
        """Return top-left corner of location where contained, held object will be drawn given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        factor = (1-self.holding_scale) + (1-self.container_scale)/2*self.holding_scale
        return tuple((np.asarray(scaled_loc) + self.scale*factor).astype(int))


    def on_cleanup(self):
        pygame.quit()