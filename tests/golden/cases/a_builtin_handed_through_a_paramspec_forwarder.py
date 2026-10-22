# WHAT: a builtin function and a class handed to a ParamSpec forwarder
# (`fn: Callable[P, R]` with `*args: P.args`), and a builtin the emitter
# answers only at its call (`repr`, `min`) given to a Callable parameter.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: `call_with(print, "hi")`
# compiled and was then refused by the lowering ("cannot adapt
# builtins.function"), a class was refused by the solver, and `repr` as a
# value was an unresolved name; each now reaches its body through a
# forwarder, and that it reached the right one is seen in what it prints.
from collections.abc import Callable


class Box:
    def __init__(self, item: str) -> None:
        self.item = item
        print("boxed", item)


def call_with[**P](fn: Callable[P, object], *args: P.args, **kwargs: P.kwargs) -> object:
    return fn(*args, **kwargs)


def show(f: Callable[[str], str], value: str) -> str:
    return f(value)


def fold(f: Callable[[list[int]], int], values: list[int]) -> int:
    return f(values)


call_with(print, "hi", "there")
call_with(Box, "x")
print(call_with(int, "ff", 16))
print(call_with(divmod, 7, 2))
try:
    call_with(int, "nope")
except ValueError as e:
    print("ValueError:", e)
print(show(repr, "q"))
print(fold(min, [3, 1, 2]), fold(max, [3, 1, 2]), fold(sum, [3, 1, 2]))
