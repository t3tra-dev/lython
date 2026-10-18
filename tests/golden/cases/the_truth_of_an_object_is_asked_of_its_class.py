# WHAT: `bool(x)`, `not x`, `if x:` and the truth tests in `any`/`all`/a
# comprehension filter on a value held as `object` answer as CPython's
# PyObject_IsTrue does: None and False are false, numbers by zero, containers
# and strings by length, a class by its `__bool__` (inherited too, and ahead
# of a `__len__`), else by its `__len__` with the length checks, else true.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the class is the one the
# value has at run time; the static type says only `object`. Every truth
# test here was refused ("builtins.object.__bool__ is declared by the
# standard-library contract but has no implementation").
class Box:
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n


class Sub(Box):
    pass


class Base:
    def __bool__(self) -> bool:
        return False


class Kid(Base):
    def __len__(self) -> int:
        return 5


class Flag:
    def __init__(self, on: bool) -> None:
        self.on = on

    def __bool__(self) -> bool:
        return self.on


class Plain:
    pass


def truthy(x: object) -> bool:
    return bool(x)


def truth(x: object) -> str:
    try:
        return "T" if x else "F"
    except (ValueError, OverflowError) as e:
        return type(e).__name__


print(truthy(0), truthy(1), truthy(0.0), truthy(2.5), truthy(0j), truthy(1j))
print(truthy(""), truthy("a"), truthy(b""), truthy(b"x"), truthy(None))
print(truthy([]), truthy([0]), truthy(()), truthy((0,)), truthy({}),
      truthy({"k": 0}), truthy(set()), truthy({0}), truthy(range(0)),
      truthy(range(3)))
print(truthy(True), truthy(False), truthy(Flag(False)), truthy(Flag(True)),
      truthy(Plain()), truthy(ValueError()), truthy(int))
objs: list[object] = [Box(0), Box(1), Sub(0), Sub(2), Kid(), Plain(),
                      Box(-1), Box(2 ** 70), 0, -0.0, 1e-300, "", "x"]
print([truth(o) for o in objs])
print([type(o).__name__ for o in objs[:6] if o])
print(sum(1 for o in objs[8:] if not o), any(objs[8:10]), all(objs[10:]),
      all(objs[11:]))
