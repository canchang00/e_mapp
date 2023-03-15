from __future__ import annotations

from typing import Callable,Optional,TypeVar,Union

from copy import deepcopy

from algorithm.parser.ast.node import *
from algorithm.parser.ast.tree import *


from algorithm.parser.tokenizer import lexer


def first(*first: str):
    def decorator(f: Callable[[Parser], T]) -> Rule[T]:
        f.first = frozenset(first)
        return f

    return decorator


@first("while")
def p_while(self:Parser)-> While:
    """
    while : 'while' '(' execution ')' '{' statement_list '}'
    """
    lookahead = self.lookahead
    lookahead("while")
    lookahead("LPAREN")
    if self.next=="True":
        lookahead("True")
        cond=Execution("Tautology",[],self.next_token.lineno)
    else:
        cond = p_execution(self)
    lookahead("RPAREN")
    lookahead("LBRACE")
    body = p_block(self)
    lookahead("RBRACE")
    return While(cond, body)

@first("repeat")
def p_repeat(self:Parser):
    """
    repeat : 'repeat' '{' statement_list '}'
    """
    lookahead=self.lookahead
    lookahead("repeat")
    lookahead("LBRACE")
    body=p_block(self)
    lookahead("RBRACE")
    return Repeat(body)
    

@first("ID")
def p_execution(self:Parser)-> Execution:
    lookahead=self.lookahead
    lineno=self.next_token.lineno
    if self.next=="ID":
        ident = lookahead("ID")
        lookahead("LPAREN")
        follow=["RPAREN"]
        args=[]
        while self.next not in follow:
            arg= lookahead("ID")
            args.append(arg)
        lookahead("RPAREN")
        return Execution(ident,args,lineno)
    else:
        return None
        


def p_block(self: Parser) -> Block:
    "block : (statement | declaration ';')*"

    def p_block_item(self: Parser):
        if self.next in p_while.first:
            return p_while(self)
        elif self.next in p_if.first:
            return p_if(self)
        elif self.next in p_parallel.first:
            return p_parallel(self)
        elif self.next in p_alternative.first:
            return p_alternative(self)
        elif self.next in p_repeat.first:
            return p_repeat(self)
        else:                               # a subroutine
            return p_execution(self)

    block = Block()
    follow = {"RBRACE"}

    while self.next not in follow:
        block_item = p_block_item(self)
        if block_item is not None:
            block.children.append(block_item)

    return block


@first("if")
def p_if(self: Parser) -> If:
    "if : 'if' '(' expression ')' statement ( 'else' statement )?"
    lookahead = self.lookahead
    lookahead("if")
    lookahead("LPAREN")
    cond = p_execution(self)
    lookahead("RPAREN")
    lookahead("LBRACE")
    body = p_block(self)
    lookahead("RBRACE")
    if self.next == "else":
        lookahead("else")
        lookahead("LBRACE")
        otherwise = p_block(self)
        lookahead("RBRACE")
        return If(cond, body, otherwise)
    return If(cond, body)

@first("parallel")
def p_parallel(self:Parser)-> Parallel:
    lookahead = self.lookahead
    lookahead("parallel")
    lookahead("LBRACE")
    
    follow=["RBRACE"]
    blocks=[]
    while self.next not in follow:
        lookahead("LBRACE")
        block=p_block(self)
        lookahead("RBRACE")
        blocks.append(block)
    lookahead("RBRACE")
    return Parallel(blocks)

@first("alternative")
def p_alternative(self:Parser)-> Parallel:
    """
    alternative : 'parallel' '(' execution ')' '{' statement_list '}'
    """
    lookahead = self.lookahead
    lookahead("alternative")
    lookahead("LBRACE")
    
    follow=["RBRACE"]
    blocks=[]
    while self.next not in follow:
        lookahead("LBRACE")
        block=p_block(self)
        lookahead("RBRACE")
        blocks.append(block)
    lookahead("RBRACE")
    return Alternative(blocks)

def p_program(self: Parser) -> Program:
    "program : Identifier '(' ')' '{' block '}'"
    lookahead = self.lookahead
    ident = lookahead("ID")
    lookahead("LPAREN")
    lookahead("RPAREN")
    lookahead("LBRACE")
    body = p_block(self)
    lookahead("RBRACE")
    tail = lookahead()
    if tail is not None:
        raise SyntaxError(f"Unexpected token {tail}")
    return Program(Function(ident, body))


class Parser:
    def __init__(self, _lexer: Optional[Lexer] = None) -> None:
        self.lexer = _lexer or lexer
        self.next_token: Optional[LexToken]

    def lookahead(self, type: Optional[str] = None) -> Any:
        tok = self.next_token
        if tok is None:
            return tok
        if tok.type == type or type is None:
            try:
                self.next_token = next(self.lexer)
            except StopIteration:
                self.next_token = None
            return tok and tok.value
        
    @property
    def next(self):
        if self.next_token is None:
            raise StopIteration
        return self.next_token.type

    # def parse(self, input: str, lexer: Optional[Lexer] = None):
    def parse(self):
        # if lexer:
        #     self.lexer = lexer
        # self.lexer.input(input)
        self.next_token = next(self.lexer)
        # try:
        return p_program(self)
        # except DecafSyntaxError as e:
        #     self.error_stack.append(e)


