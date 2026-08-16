# What this pins: an unannotated lambda bound to a NAME takes its parameter
# types from the calls that follow it. It has no type of its own -- "lambda
# requires a Callable annotation because its type contains unresolved Unknown"
# -- and the callee-position repair reads the parameters off the arguments,
# which an ASSIGNMENT does not have. The call does, and it is in the same
# suite, so the forward scan that decides an empty literal's element type
# answers this one on the same terms.
#
# Why this needs to run rather than assert on a diagnostic: the parameter type
# is what the BODY is compiled against. `f = lambda v: v * 2` reached by `f(5)`
# has to multiply an int and by `f(1.5)` a float, and `v * 2` compiles either
# way -- only the printed 10 against 3.0 says which one was chosen.
#
# ⛔ Two boundaries are here as the control, because the scan must DECLINE
# rather than guess: calls that disagree on the argument types leave the lambda
# unannotated (a body emitted at one call's types and used at another's would
# be a wrong program, not a refused one), and a name that is PASSED rather than
# called has no arguments to read. Both are recorded in
# tests/probe/wb_grid_leftovers_2026_08_16.py.
#
# Every expected line is python3.14's.

# --- the plain shapes ------------------------------------------------------
double = lambda v: v * 2
print(double(5))

halve = lambda v: v / 2
print(halve(5.0))

add = lambda a, b: a + b
print(add(1, 2), add(3, 4))

shout = lambda s: s.upper() + "!"
print(shout("hi"))


# --- inside a function, and through a local the call mentions -------------
def run(n: int) -> int:
    step = lambda v: v + 1
    total = 0
    i = 0
    while i < n:
        total = step(total)
        i += 1
    return total


print(run(4), run(0))


# --- the same lambda called more than once, agreeing ----------------------
def twice(n: int) -> int:
    scale = lambda v: v * 3
    return scale(n) + scale(n + 1)


print(twice(2))


# --- an annotated one still wins ------------------------------------------
from typing import Callable

annotated: Callable[[int], int] = lambda v: v * 10
print(annotated(4))


# --- and a lambda applied directly is unchanged ---------------------------
print((lambda v: v - 1)(9))
