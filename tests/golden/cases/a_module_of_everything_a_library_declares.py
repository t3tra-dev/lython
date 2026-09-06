# Helper for a_library_is_read_through_every_channel_at_once. One module
# holding every declaration kind that had to learn to cross a file boundary:
# a container constant, a callable constant, an enum, a class with a class
# attribute, a container class attribute, a property with a setter, a
# staticmethod, an overridable method, and an `__init_subclass__` that takes a
# class-header keyword.
from enum import Enum
from typing import Callable, Iterator

NAMES: list[str] = ["lib"]
SCALE: Callable[[int], int] = lambda n: n * 2


class Color(Enum):
    RED = 1
    BLUE = 2


class Shape:
    kind: str = "shape"
    seen: list[str] = []

    def __init__(self, size: int) -> None:
        self._size = size

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, n: int) -> None:
        self._size = n

    @staticmethod
    def sides() -> int:
        return 0

    def name(self) -> str:
        return "shape"

    @classmethod
    def __init_subclass__(cls, tag: str = "none") -> None:
        Shape.seen.append(cls.__name__ + ":" + tag)
