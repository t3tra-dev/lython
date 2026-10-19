# A builtin function passed where a callable value is expected -- an
# argument, an annotated variable, a dict value, a return -- becomes a
# function typed by the callable it is handed to. It reached the lowering as
# the builtin's name with no object ("cannot adapt builtins.function to
# runtime input 0"), while builtin call positions (`sorted(key=len)`) never
# needed one.
# Why run it: each wrapper has to call the builtin the name meant, with the
# overload the declared parameter selects (`abs` of an int and of a float).
import math
from collections.abc import Callable


def apply(f: Callable[[str], object], text: str) -> None:
    f(text)


def measure(f: Callable[[str], int], text: str) -> int:
    return f(text)


def pick() -> Callable[[str], int]:
    return len


apply(print, "printed through a callable")
print(measure(len, "four"), pick()("seven!!"))
whole: Callable[[int], int] = abs
real: Callable[[float], float] = abs
print(whole(-3), real(-2.5))
ops: dict[str, Callable[[float], float]] = {"sqrt": math.sqrt, "neg": lambda x: -x}
print(ops["sqrt"](16.0), ops["neg"](1.5))
print(sorted(["ccc", "a", "bb"], key=len))
