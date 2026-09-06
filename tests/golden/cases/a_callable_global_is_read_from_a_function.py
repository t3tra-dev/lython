# What: a module-level name annotated `Callable` is storage, so a function can
# read it. Only a CONTRACT annotation got a cell, and a callable was left
# "value-bound" -- usable at module scope and invisible one line inside a
# function, which is where a policy hook is called from:
#
#     DOUBLE: Callable[[int], int] = lambda n: n * 2
#     def use(n: int) -> int: return DOUBLE(n)   # unresolved name 'DOUBLE'
#
# ⭐ And ANNOTATING made it worse, which is what says the annotation was the
# cause: `DOUBLE = double` without one already worked, through the alias
# binding, and adding `: Callable[[int], int]` took it away.
#
# The rebinding is the cell's own benefit and needs running to see: `global`
# writes it and the reader that ran before the write and the one that ran after
# have to disagree.
from typing import Callable

import a_module_of_callable_constants as policy


def double(n: int) -> int:
    return n * 2


def triple(n: int) -> int:
    return n * 3


DOUBLE: Callable[[int], int] = lambda n: n * 2
ALIAS: Callable[[int], int] = double
PICK: Callable[[int], int] = double


def use_lambda(n: int) -> int:
    return DOUBLE(n)


def use_alias(n: int) -> int:
    return ALIAS(n)


def use_pick(n: int) -> int:
    return PICK(n)


def swap() -> None:
    global PICK
    PICK = triple


def nested(n: int) -> int:
    def inner(k: int) -> int:
        return DOUBLE(k)

    return inner(n)


print("lambda", use_lambda(3), "alias", use_alias(3), "nested", nested(4))
print("before", use_pick(3))
swap()
print("after", use_pick(3))
print("imported", policy.SCALE(3), policy.apply(4))
