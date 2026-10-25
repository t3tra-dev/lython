# Why execution: the leak gate runs this program and counts what it leaves
# allocated; every int here is wider than a machine word, so each choice,
# field store and call below moves a real int object rather than an i64.
#
# An int choice between two operator answers whose product leaves the word,
# between an operator answer and a parameter, between a list element and a
# loop-carried int, and between two fields; int fields read for arithmetic and
# stored back; a closure over a wide int; one element read once as an object.
from typing import Callable

BIG = 2 ** 70


def operator_choices(n: int, scale: int) -> int:
    t = 0
    for i in range(n):
        a = scale * (i + 1)
        t += max(a * a, a * a * 2) - min(a * a + 1, a - 1)
    return t


def clipped(xs: list[int], scale: int, hi: int) -> int:
    t = 0
    for x in xs:
        t += min(x * scale, hi)
    return t


def running_max(xs: list[int]) -> int:
    best = xs[0]
    for v in xs:
        if v > best:
            best = v
    return best


class Pair:
    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b


def field_traffic(n: int) -> int:
    p = Pair(BIG, BIG * 2)
    for i in range(n):
        p.a = p.a + p.b % 1000 + i
        p.b = p.a if p.a % 3 == 0 else p.b
    return (p.a - BIG) * 1000 + (p.b - BIG) % 1000


def make_adder(k: int) -> Callable[[int], int]:
    def add(x: int) -> int:
        return x + k
    return add


def closure_calls(n: int) -> int:
    f = make_adder(BIG)
    t = BIG
    for _ in range(n):
        t = f(t) % (BIG * 3 + 1)
    return t


def read_once(xs: list[int]) -> str:
    out = ""
    for i in range(len(xs)):
        v = xs[i]
        out = out + str(v)[-3:]
    return out


xs = [BIG + (i * 7919) % 1009 for i in range(50)]
print(operator_choices(20, 2 ** 40))
print(clipped([1, 2, 3, 4], 2 ** 62, 2 ** 63))
print(running_max(xs) - BIG)
print(field_traffic(10))
print(closure_calls(10))
print(read_once(xs[:5]))
