# What this pins: a lambda argument to a GENERIC callable, where the type
# parameter its body needs is decided by a different argument.
#
#     functools.reduce(lambda a, b: a + b, [1, 2, 3])
#     # static type !py.typevar<"T"> does not provide manifest method '__add__'
#
# `reduce[T](function: Callable[[T, T], T], sequence: list[T])` binds T from
# the SECOND argument, and arguments are emitted left to right against the
# contract as written -- so the lambda's body was compiled against T itself.
# The call then specialized correctly, which is how the second diagnostic could
# already name `Callable[[int, int], int]`: the types were known, just not yet
# when the body needed them. Binding the parameters from the arguments that can
# be typed WITHOUT being emitted, before the walk starts, is what this fixes.
#
# Why this needs to run rather than assert on a diagnostic: the substituted
# parameter type is what the BODY compiles against, and `a + b` compiles for
# int, float, str and list alike. Only the printed value says which one the
# lambda was given -- 24 against 4.0 against "abbc" against [1, 2].
#
# ⛔ `reduce(f, seq, initial)` with a DIFFERENTLY typed initial stays refused
# ("lambda body is not compatible with its Callable annotation"). functools.py
# records why: the port has typeshed's `Callable[[T, T], T]` overload only, and
# the other one needs overload selection on an omitted argument. An initial of
# the ELEMENT type works and is below.
#
# Every expected line is python3.14's.

import functools

# --- the four element types, same lambda body ------------------------------
print(functools.reduce(lambda a, b: a + b, [1, 2, 3]))
print(functools.reduce(lambda a, b: a * b, [1, 2, 3, 4]))
print(functools.reduce(lambda a, b: a + b, [1.5, 2.5]))
print(functools.reduce(lambda a, b: a + b, ["a", "bb", "c"]))

nested: list[list[int]] = [[1], [2]]
print(functools.reduce(lambda a, b: a + b, nested))


# --- a body that is not an operator ----------------------------------------
print(functools.reduce(lambda a, b: a if a > b else b, [3, 9, 2]))
print(functools.reduce(lambda a, b: a * 10 + b, [1, 2, 3]))


# --- an initial of the element type ----------------------------------------
print(functools.reduce(lambda a, b: a + b, [1, 2, 3], 10))
print(functools.reduce(lambda a, b: a + b, ["a", "b"], "-"))


# --- THE CONTROL: an ANNOTATED callable, which always worked ---------------
from typing import Callable

named: Callable[[int, int], int] = lambda a, b: a + b
print(functools.reduce(named, [1, 2, 3]))


# --- bisect's insort, which the docstring said could not be CALLED ---------
# "inserting into a caller-owned list through a parameter needs borrowed-
# container mutation, which the ownership layer rejects" -- it does not any
# more, and nothing pinned that it had started working.
import bisect

xs = [1, 3, 5]
bisect.insort_left(xs, 2)
bisect.insort_right(xs, 3)
print(xs)
print(bisect.bisect_left(xs, 3), bisect.bisect_right(xs, 3))

words = ["aa", "cc"]
bisect.insort_left(words, "bb")
print(words)
