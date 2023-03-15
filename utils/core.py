import recipe_planner.utils as recipe

# helpers
import numpy as np
import copy
import random
from termcolor import colored as color
from itertools import combinations
from collections import namedtuple


# -----------------------------------------------------------
# GRIDSQUARES
# -----------------------------------------------------------
GridSquareRepr = namedtuple("GridSquareRepr", "name location holding")


chop_step=2

class Rep:
    FLOOR = 'f'
    COUNTER = '-'
    CUTBOARD = '/'
    DELIVERY = '*'
    TOMATO = 't'
    LETTUCE = 'l'
    ONION = 'o'
    PLATE = 'p'
    DIRTYPLATE='q'
    VASE='v'
    FIRE='h'
    FIREEXTINGUISHER='e'
    PAN='n'
    STOVE='s'
    DISHWASHER='w'
    TOMATOSUPPLY='T'
    ONIONSUPPLY='O'
    LETTUCESUPPLY='L'
    RUBBISHBIN='R'

class GridSquare:
    def __init__(self, name, location):
        self.name = name
        self.location = location   # (x, y) tuple
        self.holding = None
        self.color = 'white'
        self.collidable = True     # cannot go through
        self.dynamic = False       # cannot move around

    def __str__(self):
        return self.rep

    def __eq__(self, o):
        return isinstance(o, GridSquare) and self.name == o.name

    def __copy__(self):
        gs = type(self)(self.location)
        gs.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            gs.holding = copy.copy(self.holding)
        return gs

    def acquire(self, obj):
        obj.location = self.location
        self.holding = obj

    def release(self):
        temp = self.holding
        self.holding = None
        return temp

class Floor(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self,"Floor", location)
        self.color = None
        self.rep = Rep.FLOOR
        self.collidable = False
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class Counter(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self,"Counter", location)
        self.rep = Rep.COUNTER
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class AgentCounter(Counter):
    def __init__(self, location):
        GridSquare.__init__(self,"Agent-Counter", location)
        self.rep = Rep.COUNTER
        self.collidable = True
    def __eq__(self, other):
        return Counter.__eq__(self, other)
    def __hash__(self):
        return Counter.__hash__(self)
    def get_repr(self):
        return GridSquareRepr(name=self.name, location=self.location, holding= None)

class Cutboard(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Cutboard", location)
        self.rep = Rep.CUTBOARD
        self.collidable = True
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)
    
class Stove(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Stove", location)
        self.rep = Rep.STOVE
        self.collidable = True
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class Dishwasher(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Dishwasher", location)
        self.rep = Rep.DISHWASHER
        self.collidable = True
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)
    
class Delivery(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Delivery", location)
        self.rep = Rep.DELIVERY
        self.holding = []
    def acquire(self, obj):
        obj.location = self.location
        self.holding.append(obj)
    def release(self):
        if self.holding:
            return self.holding.pop()
        else: return None
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class TomatoSupply(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "TomatoSupply", location)
        self.rep = Rep.TOMATOSUPPLY
        self.holding = []
    def acquire(self, obj):
        obj.location = self.location
        self.holding.append(obj)
    def release(self):
        if self.holding:
            return self.holding.pop()
        else: return None
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)
    
class OnionSupply(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "OnionSupply", location)
        self.rep = Rep.ONIONSUPPLY
        self.holding = []
    def acquire(self, obj):
        obj.location = self.location
        self.holding.append(obj)
    def release(self):
        if self.holding:
            return self.holding.pop()
        else: return None
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)
    
class LettuceSupply(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "LettuceSupply", location)
        self.rep = Rep.LETTUCESUPPLY
        self.holding = []
    def acquire(self, obj):
        obj.location = self.location
        self.holding.append(obj)
    def release(self):
        if self.holding:
            return self.holding.pop()
        else: return None
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class RubbishBin(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "RubbishBin", location)
        self.rep = Rep.RUBBISHBIN
        self.holding = []
    def acquire(self, obj):
        obj.location = self.location
        self.holding.append(obj)
    def release(self):
        if self.holding:
            return self.holding.pop()
        else: return None
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)
    
    
# -----------------------------------------------------------
# OBJECTS
# -----------------------------------------------------------
# Objects are wrappers around foods items, plates, and any combination of them

ObjectRepr = namedtuple("ObjectRepr", "name location is_held done_percent")

class Object:
    def __init__(self, location, contents):
        self.location = location
        self.contents = contents if isinstance(contents, list) else [contents]
        self.is_held = False
        self.update_names()
        self.collidable = False
        self.dynamic = False
        self.done_percent = 0.0
    
    def is_done(self):
        return self.done_percent>=1.0

    def __str__(self):
        res = "".join(list(map(lambda x : str(x), sorted(self.contents, key=lambda i: str(i)))))
        if res.endswith("p"):
            res="p"+res[:-1]
        return res

    def __eq__(self, other):
        # check that content is the same and in the same state(s)
        return isinstance(other, Object) and \
                self.name == other.name and \
                len(self.contents) == len(other.contents) and \
                self.full_name == other.full_name
                # all([i == j for i, j in zip(sorted(self.contents, key=lambda x: x.name),
                #                             sorted(other.contents, key=lambda x: x.name))])

    def __copy__(self):
        new = Object(self.location, self.contents[0])
        new.__dict__ = self.__dict__.copy()
        new.contents = [copy.copy(c) for c in self.contents]
        return new

    def get_repr(self):
        return ObjectRepr(name=self.full_name, location=self.location, is_held=self.is_held,done_percent=self.done_percent)

    def update_names(self):
        # concatenate names of alphabetically sorted items, e.g.
        sorted_contents = sorted(self.contents, key=lambda c: c.name)
        self.name = "-".join([c.name for c in sorted_contents])
        self.full_name = "-".join([c.full_name for c in sorted_contents])

    def contains(self, c_name):
        return c_name in list(map(lambda c : c.name, self.contents))

    def needs_chopped(self):
        if len(self.contents) > 1: return False
        return self.contents[0].needs_chopped()
    
    def needs_cooked(self):
        for idx,obj in enumerate(self.contents):
            if obj.needs_cooked():
                return True
        return False

    def is_chopped(self):
        for c in self.contents:
            if isinstance(c, Plate) or c.get_state() != 'Chopped':
                return False
        return True

    def chop(self):
        assert len(self.contents) == 1
        assert self.needs_chopped()
        self.done_percent+=chop_step
        self.contents[0].done_percent=self.done_percent
        if self.is_done():
            self.contents[0].update_state()
            assert not (self.needs_chopped())
            self.update_names()
            
    def cook(self):
        assert self.needs_cooked()
        for idx,obj in enumerate(self.contents):
            if isinstance(obj,Plate):
                continue
            elif obj.needs_cooked():
                self.contents[idx].update_state()
        self.update_names()
                
        
    def chop2pieces(self):
        assert len(self.contents) == 1
        assert self.needs_chopped()
        self.done_percent=1.1
        self.contents[0].done_percent=self.done_percent
        if self.is_done():
            self.contents[0].update_state()
            assert not (self.needs_chopped())
            self.update_names()        

    def merge(self, obj):
        if isinstance(obj, Object):
            # move obj's contents into this instance
            for i in obj.contents: self.contents.append(i)
        elif not (isinstance(obj, Food) or isinstance(obj, Plate)):
            raise ValueError("Incorrect merge object: {}".format(obj))
        else:
            self.contents.append(obj)
        self.update_names()

    def unmerge(self, full_name):
        # remove by full_name, assumming all unique contents
        matching = list(filter(lambda c: c.full_name == full_name, self.contents))
        self.contents.remove(matching[0])
        self.update_names()
        return matching[0]

    def is_merged(self):
        return len(self.contents) > 1

    def is_deliverable(self):
        # must be merged, and all contents must be Plates or Foods in done state
        for c in self.contents:
            if not (isinstance(c, Plate) or (isinstance(c, Food) and c.done())):
                return False
        return (self.is_merged() and 'p' in str(self))


def mergeable(obj1, obj2):
    # No shared objects
    contents1_list=[str(c)[-1] for c in obj1.contents]
    contents2_list=[str(c)[-1] for c in obj2.contents]
    if len(set(contents1_list)&set(contents2_list))>0:
        return False
    
    # No distinct states
    contents1_list=[str(c)[:-1] for c in obj1.contents]
    contents2_list=[str(c)[:-1] for c in obj2.contents]
    if len(set(contents1_list)|set(contents2_list))>2:
        return False
    
    # unmergeable objects
    unmergeables=['q','e','h','v']
    if str(obj1) in unmergeables or str(obj2) in unmergeables:
        return False
    
    contents = obj1.contents + obj2.contents
    # check that there is at most one plate
    try:
        contents.remove(Plate())
    except:
        pass  # do nothing, 1 plate is ok
    finally:
        try:
            contents.remove(Plate())
        except:
            for c in contents:   # everything else must be in last state
                if c.state_index==0:
                    return False
        else:
            return False  # more than 1 plate
    return True


# -----------------------------------------------------------

class FoodState:
    FRESH = globals()['recipe'].__dict__['Fresh']
    CHOPPED = globals()['recipe'].__dict__['Chopped']
    COOKED= globals()['recipe'].__dict__['Cooked']

class FoodSequence:
    FRESH = [FoodState.FRESH]
    FRESH_CHOPPED = [FoodState.FRESH, FoodState.CHOPPED]
    FRESH_CHOPPED_COOKED = [FoodState.FRESH, FoodState.CHOPPED, FoodState.COOKED]

class Food:
    def __init__(self):
        self.state = self.state_seq[self.state_index]
        self.movable = False
        self.color = self._set_color()
        self.update_names()
        self.done_percent = 0.

    def __str__(self):
        return color(self.rep, self.color)

    # def __hash__(self):
    #     return hash((self.state, self.name))

    def __eq__(self, other):
        return isinstance(other, Food) and self.get_state() == other.get_state()

    def __len__(self):
        return 1   # one food unit

    def set_state(self, state):
        assert state in self.state_seq, "Desired state {} does not exist for the food with sequence {}".format(state, self.state_seq)
        self.state_index = self.state_seq.index(state)
        self.state = state
        self.update_names()

    def get_state(self):
        return self.state.__name__

    def update_names(self):
        self.full_name = '{}{}'.format(self.get_state(), self.name)

    
    def needs_chopped(self):
        return self.state_seq[(self.state_index+1)%len(self.state_seq)] == FoodState.CHOPPED

    def needs_cooked(self):
        return self.state_seq[(self.state_index+1)%len(self.state_seq)] == FoodState.COOKED
    
    def done(self):
        # return (self.state_index % len(self.state_seq)) == len(self.state_seq) - 1
        return (self.state_index % len(self.state_seq))!=0

    def update_state(self):
        self.state_index += 1
        assert 0 <= self.state_index and self.state_index < len(self.state_seq), "State index is out of bounds for its state sequence"
        self.state = self.state_seq[self.state_index]
        self.update_names()

    def _set_color(self):
        pass

class Tomato(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED_COOKED
        self.rep = 't'
        self.name = 'Tomato'
        Food.__init__(self)
    def __hash__(self):
        return Food.__hash__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __str__(self):
        if self.state_index==0:
            return 't'
        elif self.state_index==1:
            return 'Ct'
        elif self.state_index==2:
            return 'Dt'

class Lettuce(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED_COOKED
        self.rep = 'l'
        self.name = 'Lettuce'
        Food.__init__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __hash__(self):
        return Food.__hash__(self)
    def __str__(self):
        if self.state_index==0:
            return 'l'
        elif self.state_index==1:
            return 'Cl'
        elif self.state_index==2:
            return 'Dl'

class Onion(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED_COOKED
        self.rep = 'o'
        self.name = 'Onion'
        Food.__init__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __hash__(self):
        return Food.__hash__(self)
    def __str__(self):
        if self.state_index==0:
            return 'o'
        elif self.state_index==1:
            return 'Co'
        elif self.state_index==2:
            return 'Do'


# -----------------------------------------------------------

class Plate:
    def __init__(self):
        self.rep = "p"
        self.name = 'Plate'
        self.full_name = 'Plate'
        self.color = 'white'
    def __hash__(self):
        return hash((self.name))
    def __str__(self):
        return color(self.rep, self.color)
    def __eq__(self, other):
        return isinstance(other, Plate)
    def __copy__(self):
        return Plate()
    def needs_chopped(self):
        return False
    def needs_cooked(self):
        return False
    def __str__(self):
        return 'p'
    
class Dirtyplate:
    def __init__(self):
        self.rep = "q"
        self.name = 'Dirtyplate'
        self.full_name = 'Dirtyplate'
        self.color = 'white'
    def __hash__(self):
        return hash((self.name))
    def __eq__(self, other):
        return isinstance(other, Dirtyplate)
    def __copy__(self):
        return Dirtyplate()
    def needs_chopped(self):
        return False
    def needs_cooked(self):
        return False
    def __str__(self):
        return 'q'
    
class Vase:
    def __init__(self):
        self.rep = "v"
        self.name = 'vase'
        self.full_name = 'vase'
        self.color = 'white'
    def __hash__(self):
        return hash((self.name))
    def __eq__(self, other):
        return isinstance(other, Vase)
    def __copy__(self):
        return Vase()
    def needs_chopped(self):
        return False
    def needs_cooked(self):
        return False
    def __str__(self):
        return 'v'
    
class Fire:
    def __init__(self):
        self.rep = "h"
        self.name = 'fire'
        self.full_name = 'fire'
        self.color = 'white'
    def __hash__(self):
        return hash((self.name))
    def __eq__(self, other):
        return isinstance(other, Fire)
    def __copy__(self):
        return Fire()
    def needs_chopped(self):
        return False
    def needs_cooked(self):
        return False
    def __str__(self):
        return 'h'
    
class FireExtinguisher:
    def __init__(self):
        self.rep = "e"
        self.name = 'extinguisher'
        self.full_name = 'extinguisher'
        self.color = 'white'
    def __hash__(self):
        return hash((self.name))
    def __eq__(self, other):
        return isinstance(other, FireExtinguisher)
    def __copy__(self):
        return FireExtinguisher()
    def needs_chopped(self):
        return False
    def needs_cooked(self):
        return False
    def __str__(self):
        return 'e'

# class Pan:
#     def __init__(self):
#         self.rep = "n"
#         self.name = 'Pan'
#         self.full_name = 'Pan'
#         self.color = 'white'
#     def __hash__(self):
#         return hash((self.name))
#     def __str__(self):
#         return color(self.rep, self.color)
#     def __eq__(self, other):
#         return isinstance(other, Pan)
#     def __copy__(self):
#         return Pan()
#     def needs_chopped(self):
#         return False
#     def __str__(self):
#         return 'n'

# -----------------------------------------------------------
# PARSING
# -----------------------------------------------------------
RepToClass = {
    Rep.FLOOR: globals()['Floor'],
    Rep.COUNTER: globals()['Counter'],
    Rep.CUTBOARD: globals()['Cutboard'],
    Rep.DELIVERY: globals()['Delivery'],
    Rep.TOMATO: globals()['Tomato'],
    Rep.LETTUCE: globals()['Lettuce'],
    Rep.ONION: globals()['Onion'],
    Rep.PLATE: globals()['Plate'],
    Rep.DIRTYPLATE:globals()['Dirtyplate'],
    Rep.FIRE: globals()['Fire'],
    Rep.FIREEXTINGUISHER: globals()['FireExtinguisher'],
    Rep.VASE: globals()['Vase'],
    # Rep.PAN: globals()['Pan'],
    Rep.STOVE: globals()['Stove'],
    Rep.DISHWASHER: globals()['Dishwasher'],
    Rep.TOMATOSUPPLY:globals()['TomatoSupply'],
    Rep.ONIONSUPPLY:globals()['OnionSupply'],
    Rep.LETTUCESUPPLY:globals()['LettuceSupply'],
    Rep.RUBBISHBIN:globals()['RubbishBin'],
}



