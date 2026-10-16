# What: a `try` or a `with` inside a generator -- in its loop, around a yield,
# after one, with no yield inside it. Each of these was refused ("unwind
# cleanup cannot target a handler entry with block arguments") and recorded in
# tests/probe/wb_try_inside_a_loop_inside_a_generator.py, which an earlier
# change resolved without the probe being retired; no golden held a `with` in
# a generator's loop, so this pins all of them. Not red-checked: the fix
# predates it.
#
# Why execution: the order of enter / exit / finally against the yields is the
# behaviour, and only running shows it.
from typing import Iterator


class X:
    def __enter__(self) -> int:
        print("enter")
        return 5

    def __exit__(self, a: object, b: object, c: object) -> None:
        print("exit")


def tf(xs: list[int]) -> Iterator[int]:
    for x in xs:
        try:
            yield x
        finally:
            print("fin", x)


def te(xs: list[int]) -> Iterator[int]:
    for x in xs:
        try:
            if x == 2:
                raise ValueError("v")
            yield x
        except ValueError as e:
            print("caught", e)


def w1() -> Iterator[int]:
    with X() as n:
        yield n


def w2() -> Iterator[int]:
    with X():
        yield 1


def w3() -> Iterator[int]:
    with X() as n:
        print(n)
    yield 2


def w4() -> Iterator[int]:
    yield 0
    with X() as n:
        print(n)


def w5(xs: list[int]) -> Iterator[int]:
    for x in xs:
        with X() as n:
            yield n + x


print(list(tf([1, 2])), list(te([1, 2, 3])))
print(list(w1()), list(w2()), list(w3()), list(w4()), list(w5([1, 2])))
