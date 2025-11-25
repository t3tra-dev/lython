from __future__ import annotations

from abc import ABC

__all__ = ["prim", "i1", "i8"]

class prim(ABC):
    pass

class i1(prim):
    def __eq__(self, other: i1) -> i1:  # type: ignore
        pass

    def __ne__(self, other: i1) -> i1:  # type: ignore
        pass

class i8(prim):
    def __add__(self, other: i8) -> i8:
        pass

    def __sub__(self, other: i8) -> i8:
        pass

    def __mul__(self, other: i8) -> i8:
        pass

    def __lt__(self, other: i8) -> i1:
        pass

    def __le__(self, other: i8) -> i1:
        pass

    def __eq__(self, other: i8) -> i1:  # type: ignore
        pass

    def __ne__(self, other: i8) -> i1:  # type: ignore
        pass

    def __gt__(self, other: i8) -> i1:
        pass

    def __ge__(self, other: i8) -> i1:
        pass
