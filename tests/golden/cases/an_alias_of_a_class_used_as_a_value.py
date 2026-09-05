# What: a module-level name bound to a class, then used the way the class name
# itself can be -- constructed inside a function, captured by a nested def,
# raised, and caught. A class NAME is predeclared and emits its type object; a
# module global holding one has no storage here, so the alias was "unresolved
# name 'W'". Runtime values, because the question is which class each use
# builds or catches, and the same lines spelled with the class name have always
# worked.
from typing import Callable


class Widget:
    def __init__(self, n: int) -> None:
        self.n: int = n

    def label(self) -> str:
        return "W" + str(self.n)


class MyError(Exception):
    pass


W = Widget
E = MyError
Err = ValueError


def direct(n: int) -> str:
    cls = W
    return cls(n).label()


def captured(n: int) -> str:
    cls = W

    def build(v: int) -> str:
        return cls(v).label()

    return build(n)


def factory() -> Callable[[int], W]:
    cls = W

    def build(v: int) -> W:
        return cls(v)

    return build


def raised(n: int) -> str:
    try:
        if n < 0:
            raise E("mine")
        raise Err("builtin")
    except E as e:
        return "mine:" + str(e)
    except Err as e:
        return "builtin:" + str(e)


def annotated(w: W) -> int:
    return w.n


print(direct(1))
print(captured(2))
print(factory()(3).label())
print(raised(-1), raised(1))
print(annotated(Widget(4)))
