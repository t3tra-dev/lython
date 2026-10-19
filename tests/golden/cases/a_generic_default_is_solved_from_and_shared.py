# WHAT: generic functions and methods with defaults. A call that leaves a
# default out solves the type parameter from it, as the value the call passes
# is the default; and a default that is not a literal is ONE value for every
# call of every specialization, evaluated once at the def -- a list it
# appends to grows across calls with different type arguments.
#
# WHY THIS IS RUN: what the omitted default is, and that one object is shared,
# are run-time facts. `pick()` was refused ("this call does not determine the
# type arguments"), and a non-literal default of a generic function was read
# from a cell nothing had written: `g(1)` crashed.
calls: list[str] = []


def make() -> list[int]:
    calls.append("make")
    return [1]


def pick[E: BaseException](expected: type[E] = BaseException) -> str:
    return expected.__name__


def g[T](x: T, xs: list[int] = make()) -> int:
    xs.append(len(xs))
    return len(xs)


def label[T](x: T, *, unit: list[str] = ["items"]) -> str:
    return str(x) + " " + unit[0]


class Case:
    def pick[E: BaseException](self, expected: type[E] = BaseException) -> str:
        return expected.__name__


print(calls)
print(pick(), pick(ValueError))
print(g(1), g("a"), g(2.5), calls)
print(label(3), label("x", unit=["names"]))
print(Case().pick(), Case().pick(KeyError))
