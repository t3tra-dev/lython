# WHAT: floats in functions that have no clone to fall back to -- a loop that
# carries a float, a float read out of a list, a module-level loop -- answer
# as CPython: a zero divisor raises (and a `try` there catches it), a float
# compared with an int past 2**53 compares exactly, float() of an int too big
# for a double raises OverflowError, floats that are no immediate (1e300, nan,
# -0.0, 5e-324, inf) survive a list and a dict, and a float reaches a union
# parameter, an object parameter, a generator, a ternary, a set and an
# f-string as itself.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: a float here is an f64
# that becomes an object only where a reader needs one, and the operations the
# f64 cannot answer take a runtime call on the spot; only running every such
# reader and every such operation shows that each answer is CPython's.
from typing import Iterator


def total(xs: list[float]) -> float:
    s = 0.0
    for x in xs:
        s += x * x
    print("partial", s)
    return s


def divide_all(xs: list[float], d: float) -> list[float]:
    out: list[float] = []
    for x in xs:
        try:
            out.append(x / d)
        except ZeroDivisionError as e:
            print("ZeroDivisionError:", e)
            out.append(-1.0)
    return out


def compare(x: float, n: int) -> str:
    return f"{x < n} {x == n} {x > n} {n <= x}"


def show(v: object) -> None:
    print(v)


def maybe(v: float | None) -> str:
    return "none" if v is None else f"{v:.3f}"


def halves(n: int) -> Iterator[float]:
    x = 1.0
    for _ in range(n):
        yield x
        x = x / 2.0


def clamp(x: float, lo: float, hi: float) -> float:
    y = lo if x < lo else x
    return hi if y > hi else y


print(total([1.5, 2.5, 3.0]))
print(divide_all([1.0, 2.0], 0.0), divide_all([1.0, 3.0], 4.0))
print(compare(9007199254740992.0, 2 ** 53 + 1), compare(1e300, 10 ** 300))
print(compare(float("nan"), 0), compare(-0.0, 0))
try:
    print(float(10 ** 400) * 1.0)
except OverflowError as e:
    print("OverflowError:", e)
odd = [1e300 * 1.0, float("nan") * 1.0, -0.0 * 1.0, 5e-324 * 1.0, float("inf") * 2.0]
table: dict[int, float] = {}
for i, v in enumerate(odd):
    table[i] = v * 1.0
print(odd, table)
show(2.5 * 2.0)
print(maybe(1.0 / 3.0), maybe(None))
print(list(halves(4)))
print(clamp(-1.0, 0.0, 1.0), clamp(0.25, 0.0, 1.0), clamp(7.0, 0.0, 1.0))
seen = {0.5 * 2.0, 1.0, 2.0 / 2.0}
print(seen, 1.0 in seen)
acc = 0.0
for k in range(1000):
    acc = acc + k * 0.001
print(acc, f"{acc:.6e}", repr(acc), str(-acc))
