# What this pins: a nested function that captures a FUNCTION-typed parameter.
# Capturing an int worked; capturing a callable was refused in the lowering:
#
#     def wrap(fn: Callable[[int], int]) -> Callable[[int], int]:
#         def inner(n: int) -> int:
#             return fn(n)          # function target wrap$inner closure 0 has
#         return inner              # contract 'builtins.function', expected
#                                   # '!py.callable<[int], returns=[int]>'
#
# A function value has one physical shape, `builtins.function`, which is why
# calling a callable-typed PARAMETER always worked. The closure slot's declared
# type is the emitter's precise callable, so the ABI compared the erased value
# against the precise slot and refused -- the wrapper-closure idiom, which is
# every decorator written by hand.
#
# Why this must run: the fix turns a refusal into a computed value, so what
# proves it is the answer. `twice` calls the captured function twice and
# `compose` threads two of them, which distinguishes "the right function got
# captured" from "a function got captured".
#
# ⛔ THE CAPTURE IS STILL CHECKED, at the emitter, where the signature is still
# known: `wrap(shout)` with `shout: Callable[[str], str]` is refused with "call
# arguments do not match the Callable contract". Only the lowering's re-check of
# an already-checked capture is relaxed, and only in that one direction.
#
# ⛔ AND THE CAPTURED FUNCTION MAY NOT ITSELF CARRY CAPTURES -- `add_one(double(
# base))` raises "TypeError: callable target is not available" rather than
# answering, because a function object does not carry its closure at run time.
# That is pinned next door in tests/golden/errors/nested_wrapper_closure.py.
# This relaxation was reverted once for answering 6 there instead of raising;
# what made it landable is the indirect call refusing to devirtualize on a
# target it does not statically know.
from typing import Callable


def wrap(fn: Callable[[int], int]) -> Callable[[int], int]:
    def inner(n: int) -> int:
        return fn(n)

    return inner


def twice(fn: Callable[[int], int]) -> Callable[[int], int]:
    def inner(n: int) -> int:
        return fn(fn(n))

    return inner


def compose(f: Callable[[int], int], g: Callable[[int], int]) -> Callable[[int], int]:
    def inner(n: int) -> int:
        return f(g(n))

    return inner


def bind(fn: Callable[[int, int], int], k: int) -> Callable[[int], int]:
    def inner(n: int) -> int:
        return fn(n, k)

    return inner


def decorate(fn: Callable[[str], str]) -> Callable[[str], str]:
    def inner(s: str) -> str:
        return fn(s) + "!"

    return inner


def inc(n: int) -> int:
    return n + 1


def dbl(n: int) -> int:
    return n * 2


def add(a: int, b: int) -> int:
    return a + b


def shout(s: str) -> str:
    return s.upper()


print(wrap(inc)(3), twice(inc)(3), twice(dbl)(3))
print(compose(inc, dbl)(5), compose(dbl, inc)(5))
print(bind(add, 3)(4))
print(decorate(shout)("hi"))

total = 0
i = 0
while i < 100:
    total += twice(inc)(i)
    i += 1
print(total)
