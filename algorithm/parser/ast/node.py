"""
Module that contains enums which list out all operators,
base class `Node` of all AST nodes,
and a helper type `NullType` along with its instance `NULL`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto, unique
from typing import Any, Optional, TypeVar, Union


class Node(ABC):
    """
    Base class of all AST nodes.
    """

    def __init__(self, name: str) -> None:
        """Constructor.
        `name`: name of this kind of node. Used when represents the node by a string.
        `_attrs`: used to store additional information on AST nodes.
        """
        self.name = name
        self._attrs = {}
        self.lineno=0

    @abstractmethod
    def __len__(self) -> int:
        """Returns its children count."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: int) -> Node:
        """
        Get one of its children by index.
        Not that children of a AST node are always AST nodes.
        """
        raise NotImplementedError

    def is_leaf(self):
        return False

    def setattr(self, name: str, value: Any):
        """Set additional information on AST node."""
        self._attrs[name] = value

    def getattr(self, name: str) -> Any:
        """
        Get additional information on AST node.
        Note that the default return value is `None` when the given name is not present.
        """
        return self._attrs.get(name, None)

    def __iter__(self):
        """Iterates its children."""
        for i in range(0, len(self)):
            yield self[i]

    def __bool__(self):
        """
        Used in contexts like `if`.
        Makes null-checking easier.
        """
        return True

    def __str__(self) -> str:
        """
        Recursively stringify itself and its children.
        Override this method when necesssary.
        """
        if len(self) == 0:
            return self.name

        return "{}[{}]".format(
            self.name,
            ", ".join(map(str, self)),
        )

    def __repr__(self) -> str:
        return self.__str__()


class NullType(Node):
    """
    Helper class that represents an empty node.
    You can take `If` in `.tree` as an example.
    """

    def __init__(self) -> None:
        super().__init__("NULL")

    def __getitem__(self, key: int) -> Node:
        return super().__getitem__(key)

    def __len__(self) -> int:
        return 0

    def __bool__(self):
        return False

    def is_leaf(self):
        return True

