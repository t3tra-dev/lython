# WHAT: a function and a lambda handed to a parameter typed `str |
# Callable[[int], str]` and `Callable[[int], str] | None`, each call taking
# the callable arm, beside calls that take the other arm.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the emitter accepted the
# call and the lowering refused it ("union injection cannot map
# builtins.function"): a function object names the runtime class
# `builtins.function`, which no Callable member equals. Which arm ran is seen
# in the text each call returns.
from collections.abc import Callable


def render(spec: str | Callable[[int], str], value: int) -> str:
    if isinstance(spec, str):
        return spec + str(value)
    return spec(value)


def maybe(f: Callable[[int], str] | None, value: int) -> str:
    if f is None:
        return "none"
    return f(value)


def shout(n: int) -> str:
    return "N=%d!" % n


print(render("v=", 3), render(shout, 4), render(lambda n: str(n * 2), 5))
print(maybe(shout, 6), maybe(None, 7))
