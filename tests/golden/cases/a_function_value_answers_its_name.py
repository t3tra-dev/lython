# WHAT: `f.__name__` for a function VALUE -- a parameter, a list element --
# holding a def, a lambda, or a class or builtin handed where a callable is
# expected.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which function the value
# holds is decided by the program at run time.
#
# ⛔ It was refused ("'__name__' of a function is folded at compile time");
# argparse names the `type=` in "invalid int value".
from collections.abc import Callable


def shout(s: str) -> object:
    return s.upper()


def name_of(f: Callable[[str], object]) -> str:
    return f.__name__


fs: list[Callable[[str], object]] = [int, shout, lambda s: s, float]
print([f.__name__ for f in fs])
print(name_of(shout), name_of(int))
