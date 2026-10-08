# WHAT: isinstance() against a @runtime_checkable protocol is True for exactly
# the classes that have its methods, and narrows a union to them so the
# protocol's method can be called; the same narrowing holds for an ordinary
# base when TWO members of the union derive from it (the arm used to keep the
# whole union and refuse the call for the member the test had excluded).
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the class test is a run
# time fact -- which values the loop counts and which bodies print.

from typing import Protocol, runtime_checkable

@runtime_checkable
class Closer(Protocol):
    def close(self) -> None: ...

class File:
    def close(self) -> None:
        print("closed")

class Pipe:
    def close(self) -> None:
        print("pipe closed")
    def flush(self) -> None: ...

class Rock:
    def weight(self) -> int:
        return 3

def shut(things: list[File | Pipe | Rock]) -> int:
    n = 0
    for t in things:
        if isinstance(t, Closer):
            t.close()
            n += 1
    return n

print(shut([File(), Rock(), Pipe()]))


class PlainCloser:
    def close(self) -> None: ...

class PlainFile(PlainCloser):
    def close(self) -> None:
        print("closed")

class PlainPipe(PlainCloser):
    def close(self) -> None:
        print("pipe closed")

class PlainRock:
    def weight(self) -> int:
        return 3

def shut_plain(things: list[PlainFile | PlainPipe | PlainRock]) -> int:
    n = 0
    for t in things:
        if isinstance(t, PlainCloser):
            t.close()
            n += 1
    return n

print(shut_plain([PlainFile(), PlainRock(), PlainPipe()]))
