# WHAT: `object.__repr__(x)`, `object.__str__(x)` and `object.__hash__(x)`
# answer as object's own slots do, past a class's override -- the default
# `<module.Class object at 0x...>` with the address `id` names, `repr(x)`,
# and the identity hash -- for a program class and for builtins; and the
# default repr of a value held as `object` (or in a `list[object]`) names
# the same address as `id`, as unittest's `safe_repr` relies on.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the addresses and the
# class a value has are run-time facts. `object.__repr__(a)` answered A's
# own "A()", `object.__repr__(1)` answered "1", `object.__hash__(a)` answered
# A's 5, and a boxed default repr printed the box's address.
class A:
    def __repr__(self) -> str:
        return "A()"

    def __str__(self) -> str:
        return "str A"

    def __hash__(self) -> int:
        return 5


class P:
    pass


class Bad:
    def __repr__(self) -> str:
        raise ValueError("no repr")


a = A()
p = P()
print(object.__repr__(a) == "<__main__.A object at " + hex(id(a)) + ">",
      object.__str__(a), object.__str__(p) == repr(p))
print(object.__hash__(a) != 5, object.__hash__(p) == hash(p),
      object.__hash__(a) == object.__hash__(a))


def r(o: object) -> str:
    return object.__repr__(o)


print(r(a) == object.__repr__(a), r(1)[:13], r("s")[:13], r([1])[:14],
      r(None)[:18], r(ValueError("x"))[:20])


def held(o: object) -> str:
    return repr(o)


xs: list[object] = [p]
print(repr(p) == "<__main__.P object at " + hex(id(p)) + ">",
      held(p) == repr(p), repr(xs) == "[" + repr(p) + "]")


def safe_repr(obj: object) -> str:
    try:
        return repr(obj)
    except Exception:
        return object.__repr__(obj)[:16]


print(safe_repr([1, "a"]), safe_repr(Bad()))
