# WHAT: ints in functions with no clone to fall back to -- loop counters and
# accumulators carried across a loop, values an operator answered -- answer as
# CPython when they cross 2**63 and come back, when they are stored in a list,
# a dict and a field and read back, passed to int, object and union
# parameters, returned, formatted and hashed; `//` and `%` by zero and a
# negative shift raise (and a `try` catches them, in a function and in a
# generator); a ctypes cell takes one.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: such an int is its i64
# while that is the value and the object an operator's slow arm made when it
# is not, and an object is made only where a reader needs one; only running
# every reader on both kinds shows that each answer is CPython's.
import ctypes
from typing import Iterator


def climb(n: int) -> list[int]:
    out: list[int] = []
    x = 3
    for _ in range(n):
        x = x * 1000003
        out.append(x)
    print("climbed", len(str(x)))
    return out


def fall(xs: list[int]) -> int:
    total = 0
    for x in xs:
        total = total + x % 1000 - x // (10 ** 30 + 7)
    return total


class Cell:
    def __init__(self, v: int) -> None:
        self.v = v


def show(v: object) -> str:
    return f"<{v}>"


def maybe(v: int | None) -> str:
    return "none" if v is None else str(v + 1)


def risky(a: int, b: int) -> str:
    try:
        return f"{a // b} {a % b}"
    except ZeroDivisionError as e:
        return f"ZeroDivisionError {e}"


def shifts(a: int, s: int) -> str:
    try:
        return f"{a << s} {a >> s}"
    except ValueError as e:
        return f"ValueError {e}"


def gen(n: int) -> Iterator[int]:
    acc = 1
    for i in range(1, 4):
        acc = acc * i
        yield acc // n


xs = climb(6)
print(xs[-1], fall(xs), -7 // 2, -7 % 2, 7 % -2)
table: dict[int, int] = {}
count = 0
for x in xs:
    count = count + 1
    table[x % 97] = x * count
print(sorted(table.items())[:2], len(table))
cells = [Cell(x - 1) for x in xs]
print(cells[0].v, cells[-1].v * 2)
print(show(2 ** 64 + count), maybe(count * 10 ** 20), maybe(None))
print(risky(7, 0), risky(-7, 2), risky(10 ** 40, 10 ** 20 + 1))
print(shifts(5, 70), shifts(5, -1), shifts(-(10 ** 30), 3))
print({count, count * 2 ** 70, count}, f"{count * 10 ** 25:,}", hex(count << 66))
print(True + count, count & True, (count > 3) + (count > 100))
c = ctypes.c_int(0)
c.value = count * 7
print(c.value)
for n in (2, 0):
    try:
        print(list(gen(n)))
    except ZeroDivisionError as e:
        print("gen ZeroDivisionError", e)
