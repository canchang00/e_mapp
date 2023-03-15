from typing import OrderedDict
from algorithm.parser.tokenizer import *
from algorithm.parser.symbol_table import *
from algorithm.parser.parser import Parser
import random
import copy

is_debug=True
NUM_REPEAT=5

class ProgramExecutor:
    def __init__(self, program_file,cached_program=None,cached_program_text=None):
        if cached_program is None:
            x=open(program_file).read()
            self.program_text=x
            lexer=lex.lex()
            lexer.input(x)
            parser = Parser(lexer)
            self.program=parser.parse()
        else:
            self.program=copy.deepcopy(cached_program)
            self.program_text=copy.deepcopy(cached_program_text)
        self.pointer=OrderedDict({0:self.program.mainFunc().body})
        self.stack_pointers=OrderedDict({0:[]})
        self.command_history=[]
        self.cond=OrderedDict({0:None})
        self.dependency=[]
        self.exit_program=False
        self.is_stuck=OrderedDict({0:False}) # label whether the current branch is stuck on a subroutine
        
        
    def execute_subroutine(self,subroutine,idx=None):
        if is_debug:
            if subroutine.ident=="Tautology" or "CheckDemand" or "IsOnFire":
                print("Executing subroutine:",subroutine.ident)
                self.cond[idx]=True
                return
            else:
                print("Executing subroutine:",subroutine.ident,subroutine.args,subroutine.lineno)
                return 
        raise NotImplementedError
    
    def judge_cond(self,subroutine,idx=None):
        raise NotImplementedError
    
    
    # check if the current subroutine is finished
    def is_subroutine_finished(self,):
        raise NotImplementedError
    
    def execute(self):
        while True:
            self.execute_one_step()
            if False not in self.is_stuck.values():
                print("Unfreeze all the threads")
                self.is_stuck={key:False for key in self.is_stuck.keys()}
            if self.exit_program:
                print("Exit program")
                break

        
    def execute_one_step(self,idx=None):
        hit_execution=False
        for pointer_idx in self.pointer.keys():
            if self.is_stuck[pointer_idx]:
                continue
            if self.pointer[pointer_idx] is None:
                if self.stack_pointers[pointer_idx]==[]:
                    self.exit_program=True
                else:
                    self.pointer[pointer_idx],idx=self.stack_pointers[pointer_idx][-1]
                    # print("stack pointer:",self.pointer[pointer_idx])
                    if self.pointer[pointer_idx].name=="if":
                        if idx==0:
                            if self.cond[pointer_idx] is not None:
                                if self.cond[pointer_idx]:
                                    self.stack_pointers[pointer_idx][-1]=(self.pointer[pointer_idx],1)
                                    self.pointer[pointer_idx]=self.pointer[pointer_idx][1]
                                else:
                                    if(len(self.pointer[pointer_idx])==3):
                                        self.stack_pointers[pointer_idx][-1]=(self.pointer[pointer_idx],2)
                                        self.pointer[pointer_idx]=self.pointer[pointer_idx][2]
                                    else:
                                        self.stack_pointers[pointer_idx].pop()
                                        self.pointer[pointer_idx]=None
                                self.cond[pointer_idx]=None
                            else:
                                raise NotImplementedError
                        else:
                            self.stack_pointers[pointer_idx].pop()
                            self.pointer[pointer_idx]=None
                    elif self.pointer[pointer_idx].name=="while":
                        if idx==0:
                            if self.cond[pointer_idx] is not None:
                                if self.cond[pointer_idx]:
                                    self.stack_pointers[pointer_idx][-1]=(self.pointer[pointer_idx],1)
                                    self.pointer[pointer_idx]=self.pointer[pointer_idx][1]
                                else:
                                    self.stack_pointers[pointer_idx].pop()
                                    self.pointer[pointer_idx]=None
                                self.cond[pointer_idx]=None
                            else:
                                self.judge_cond(self.pointer[pointer_idx][0])
                        elif idx==1:
                            self.stack_pointers[pointer_idx][-1]=(self.pointer[pointer_idx],0)
                            self.pointer[pointer_idx]=self.pointer[pointer_idx][0]
                    elif self.pointer[pointer_idx].name=="block":
                        if idx<len(self.pointer[pointer_idx])-1:
                            self.stack_pointers[pointer_idx][-1]=(self.pointer[pointer_idx],idx+1)
                            self.pointer[pointer_idx]=self.pointer[pointer_idx][idx+1]
                        else:
                            self.stack_pointers[pointer_idx].pop()
                            self.pointer[pointer_idx]=None
                    elif self.pointer[pointer_idx].name=="parallel":
                        if len(self.pointer)==1:
                            sp_tmp=copy.deepcopy(self.stack_pointers[pointer_idx])
                            sp_tmp.pop()
                            self.stack_pointers={0:sp_tmp}
                            self.pointer={0:None,}
                            self.is_stuck={0:False,}
                            self.cond={0:None,}
                            return False
                        elif len(self.pointer)>=2:
                            self.stack_pointers.pop(pointer_idx)
                            self.pointer.pop(pointer_idx)
                            self.is_stuck.pop(pointer_idx)
                            self.cond.pop(pointer_idx)
                            return False
                        else:
                            raise Exception("parallel error")
                    elif self.pointer[pointer_idx].name=="alternative":
                        sp_tmp=self.stack_pointers[pointer_idx]
                        sp_tmp.pop()
                        self.stack_pointers={0:sp_tmp}
                        self.pointer={0:None,}
                        self.is_stuck={0:False,}
                        return False
            elif self.pointer[pointer_idx].name=="block":
                self.stack_pointers[pointer_idx].append((self.pointer[pointer_idx],0))
                self.pointer[pointer_idx]=self.pointer[pointer_idx][0]
            elif self.pointer[pointer_idx].name=="parallel":
                self.stack_pointers[pointer_idx].append((self.pointer[pointer_idx],0))
                num_parallels=len(self.pointer[pointer_idx])
                p_tmp=copy.deepcopy(self.pointer[pointer_idx])
                max_id_plus1=max(list(self.pointer.keys()))+1
                tmp={max_id_plus1+idx:p_tmp[idx] for idx in range(num_parallels)}
                self.pointer.pop(pointer_idx)
                self.pointer={**self.pointer,**tmp}
                sp_tmp=self.stack_pointers[pointer_idx]
                self.stack_pointers.pop(pointer_idx)
                self.stack_pointers={**self.stack_pointers,**{max_id_plus1+i:copy.deepcopy(sp_tmp) for i in range(num_parallels)}}
                self.is_stuck.pop(pointer_idx)
                self.is_stuck={**self.is_stuck,**{max_id_plus1+i:False for i in range(num_parallels)}}
                self.cond.pop(pointer_idx)
                self.cond={**self.cond,**{max_id_plus1+i:None for i in range(num_parallels)}}
                return False
            elif self.pointer[pointer_idx].name=="while":
                self.stack_pointers[pointer_idx].append((self.pointer[pointer_idx],0))
                self.pointer[pointer_idx]=self.pointer[pointer_idx][0]
            elif self.pointer[pointer_idx].name=="if":
                cond=self.pointer[pointer_idx][0]
                self.stack_pointers[pointer_idx].append((self.pointer[pointer_idx],0))
                self.pointer[pointer_idx]=cond
            elif self.pointer[pointer_idx].name=="alternative":
                self.stack_pointers[pointer_idx].append((self.pointer[pointer_idx],0))
                num_alternatives=len(self.pointer[pointer_idx])
                p_tmp=self.pointer[pointer_idx]
                tmp=[p_tmp[idx] for idx in range(num_alternatives)]
                self.pointer.remove(self.pointer[pointer_idx])
                self.pointer.extend(tmp)
                sp_tmp=self.stack_pointers[pointer_idx]
                self.stack_pointers.remove(self.stack_pointers[pointer_idx])
                self.stack_pointers.extend([sp_tmp+[(p_tmp[i],0)] if sp_tmp is not None else [(p_tmp[i],0)] for i in range(num_alternatives)])
                self.is_stuck.remove(self.is_stuck[pointer_idx])
                self.is_stuck.extend([False for i in range(num_alternatives)])
                self.cond.remove(self.cond[pointer_idx])
                # TO BE COMPLETED
            elif self.pointer[pointer_idx].name=="execution":
                self.execute_subroutine(self.pointer[pointer_idx],pointer_idx)
                self.is_stuck[pointer_idx]=True
                self.pointer[pointer_idx]=None
                hit_execution=True
        return hit_execution
            
            
    def goto_next_subroutine(self,):
        while not self.execute_one_step():
            pass
            