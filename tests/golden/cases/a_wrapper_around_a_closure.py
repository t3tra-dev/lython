# WHAT: a wrapper handed a function that itself carries captures. The value
# crosses two boundaries the compiler cannot see through -- a parameter and a
# return -- and the call at the end has to reach the body the program built.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the whole question is WHICH
# body ran, and the wrong one answers. For one commit this printed 6 --
# `base(5) + 1` -- because the indirect call took the single surviving
# candidate as the answer, and a function value carrying no captures leaves
# only the zero-closure candidates standing. Then it raised "callable target is
# not available", which was honest and still not the answer.
#
# ⛔ WHAT MAKES IT ANSWER is that the captures now ride the function OBJECT:
# `LyFunction_New` writes a closure store into the word CPython calls
# `__closure__`, one box per capture, and the id dispatch reads them back for
# the target it matched.
from typing import Callable


def add_one(fn: Callable[[int], int]) -> Callable[[int], int]:
    def w(n: int) -> int:
        return fn(n) + 1

    return w


def double(fn: Callable[[int], int]) -> Callable[[int], int]:
    def w(n: int) -> int:
        return fn(n) * 2

    return w


def base(n: int) -> int:
    return n


# One level was always fine: the target is statically known there.
print(double(base)(5))

# Two levels is the case the captures are needed for.
first = double(base)
second = add_one(first)
print(second(5))

# Three, and the order the wrappers were applied in decides the answer.
third = double(second)
print(third(5))


# A capture that is not a function, reached out of a container the compiler
# cannot see through.
def adder(n: int):
    def add(x: int) -> int:
        return x + n

    return add


adders = [adder(i) for i in range(4)]
print([f(100) for f in adders])

by_name: "dict[str, Callable[[int], int]]" = {"a": adder(10), "b": adder(20)}
for key in sorted(by_name.keys()):
    print(key, by_name[key](1))

total = 0
for f in adders:
    total += f(1)
print(total)
