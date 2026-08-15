# What this pins: an int reaching a MANIFEST export's float parameter converts,
# and an int reaching a PYTHON function's float parameter does not. Both were
# "call arguments do not match the Callable contract"; only one of them should
# have been.
#
# The difference is what sits behind the parameter. `math.sqrt` is C against a
# double, so CPython converts through `__float__` at the boundary and there is
# no Python-visible parameter to keep an int in. A Python body keeps whatever
# it was handed -- the annotation is inert -- which is why `def p(x: float)`
# reached by `p(3)` is answered by emitting a second body at the int's rung
# instead (tests/probe/wb_argument_boundary_numeric_tower.py).
#
# Why this needs to run rather than assert on a diagnostic: the two rules print
# different things, and that is the whole content. `math.sqrt(16)` must be
# 4.0 and `p(3)` must be 3, not 3.0 -- a repair that converted at both
# boundaries, or at neither, compiles and gets one of them wrong.
#
# Every expected line is python3.14's.

import math

# --- an int literal, an int variable, and a bool, into float parameters ----
print(math.sqrt(16), math.sqrt(16.0))
n = 9
print(math.sqrt(n), math.sqrt(True))
print(math.fabs(-4), math.fabs(-4.5))
print(math.floor(2), math.ceil(3), math.trunc(5))
print(math.cos(0), math.sin(0), math.exp(0), math.log(1))


# --- and a PYTHON parameter keeps the int ---------------------------------
def scaled(x: float) -> float:
    return x


def described(x: float) -> str:
    return str(x)


print(scaled(3), scaled(3.0))
print(described(3), described(3.0))


class Holder:
    def take(self, x: float) -> str:
        return str(x)


print(Holder().take(3), Holder().take(3.0))
