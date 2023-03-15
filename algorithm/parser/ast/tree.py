"""
Module that defines all AST nodes.
Reading this file to grasp the basic method of defining a new AST node is recommended.
Modify this file if you want to add a new AST node.
"""

from __future__ import annotations

from typing import Any, Generic, List, Optional, TypeVar, Union
from .node import *

_T = TypeVar("_T", bound=Node)

def _index_len_err(i: int, node: Node):
    return IndexError(
        f"you are trying to index the #{i} child of node {node.name}, which has only {len(node)} children"
    )


class ListNode(Node,Generic[_T]):
    """
    Abstract node type that represents a node sequence.
    E.g. `Block` (sequence of statements).
    """

    def __init__(self, name: str, children: list) -> None:
        super().__init__(name)
        self.children = children

    def __getitem__(self, key: int) -> Node:
        return self.children.__getitem__(key)

    def __len__(self) -> int:
        return len(self.children)


class Program(ListNode["Function"]):
    """
    AST root. It should have only one children before step9.
    """

    def __init__(self, *children: Function) -> None:
        super().__init__("program", list(children))

    def functions(self) -> dict[str, Function]:
        return {func.ident: func for func in self if isinstance(func, Function)}

    def hasMainFunc(self) -> bool:
        return "main" in self.functions()

    def mainFunc(self) -> Function:
        return self.functions()["main"]


class Function(Node):
    """
    AST node that represents a function.
    """

    def __init__(
        self,
        ident,
        body: Block,
    ) -> None:
        super().__init__("function")
        self.ident = ident
        self.body = body

    def __getitem__(self, key: int) -> Node:
        return (
            self.ident,
            self.body,
        )[key]

    def __len__(self) -> int:
        return 2

class Execution(Node):
    """
    AST node that represents a function.
    """

    def __init__(self,ident: Identifier, args:List,lineno:int) -> None:
        super().__init__("execution")
        self.ident = ident
        self.args =  args
        self.lineno=lineno

    def __getitem__(self, key: int) -> Node:
        return NotImplementedError()

    def __len__(self) -> int:
        return len(self.args)
    
    def __str__(self)->str:
        return "{}({})".format(
            self.ident,
            ", ".join(map(str, self.args)),
        )

class Parallel(ListNode):
    def __init__(self,blocks) -> None:
        super().__init__("parallel", list(blocks))


    def is_block(self) -> bool:
        return True

class Alternative(ListNode):
    def __init__(self,blocks) -> None:
        super().__init__("alternative", list(blocks))


    def is_block(self) -> bool:
        return True

class If(Node):
    """
    AST node of if statement.
    """

    def __init__(
        self, cond, then, otherwise=None
    ) -> None:
        super().__init__("if")
        self.cond = cond
        self.then = then
        self.otherwise = otherwise or None

    def __getitem__(self, key: int) -> Node:
        return (self.cond, self.then, self.otherwise)[key]

    def __len__(self) -> int:
        return 3

class While(Node):
    """
    AST node of if statement.
    """

    def __init__(
        self, cond: Execution, body: Block) -> None:
        super().__init__("while")
        self.cond = cond
        self.body=body

    def __getitem__(self, key: int) -> Node:
        return (self.cond, self.body)[key]

    def __len__(self) -> int:
        return 2
    
class Repeat(Node):
    """
    AST node of if statement.
    """

    def __init__(self, body: Block) -> None:
        super().__init__("repeat")
        self.body=body

    def __getitem__(self, key: int) -> Node:
        return self.body[key]

    def __len__(self) -> int:
        return len(self.body)


class Block(ListNode):
    """
    AST node of block "statement".
    """

    def __init__(self, *children) -> None:
        super().__init__("block", list(children))


    def is_block(self) -> bool:
        return True



