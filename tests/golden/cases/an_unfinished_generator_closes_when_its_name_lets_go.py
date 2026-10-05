# What: a generator that was not run to its end is closed -- and its
# `finally` runs -- when the last name bound to it lets go of it, as in
# CPython: at the end of the frame, when the name is rebound, when a `for`
# that broke out of it or returned from inside it ends, when what holds it in
# a list or a field goes, and for a module's name at the end of the program. Not at its last use, which is before the value it
# produced is even printed.
# WHY THIS IS RUN: the `finally` prints, and when it prints relative to the
# program's own lines is the whole question.
from typing import Generator


def numbers(tag: str) -> Generator[int, None, None]:
    try:
        yield 1
        yield 2
    finally:
        print("closed", tag)


def frame_end() -> None:
    it = numbers("frame")
    print(next(it))
    print("still in frame_end")


def rebound() -> None:
    it = numbers("first")
    print(next(it))
    it = numbers("second")
    print("rebound")
    print(next(it))
    print("end of rebound")


def broke_out() -> None:
    for n in numbers("loop"):
        print("got", n)
        break
    print("after the loop")


def held_in_a_list() -> None:
    gens = [numbers("listed")]
    print(next(gens[0]))
    print("list still held")


class Holder:
    def __init__(self) -> None:
        self.gen = numbers("field")


def held_in_a_field() -> None:
    holder = Holder()
    print(next(holder.gen))
    print("holder still bound")


def returned_early(stop: bool) -> int:
    for n in numbers("returned"):
        if stop:
            return n
    return 0


frame_end()
rebound()
broke_out()
held_in_a_list()
held_in_a_field()
print("returned", returned_early(True))
module_level = numbers("module")
print(next(module_level))
print("end")
