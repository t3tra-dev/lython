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
Text = str
Num = int
Flag = bool
# The ANNOTATED spelling of the same binding: writing what the name is must not
# take the binding away.
CLS: type[Widget] = Widget


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


# A BUILTIN spelling is intercepted by name before the class-instantiation
# path -- `str(x)` is `__str__` dispatch, not construction -- so an alias of one
# has to reach that interception rather than the class binding.
def annotated_global(n: int) -> int:
    return CLS(n).n


def converted(n: int) -> str:
    return Text(n) + "/" + Text(Num("7") + 1) + "/" + Text(Flag(n))


print(direct(1))
print(captured(2))
print(factory()(3).label())
print(raised(-1), raised(1))
print(annotated(Widget(4)))
print(converted(0), converted(5))
print(annotated_global(6))
