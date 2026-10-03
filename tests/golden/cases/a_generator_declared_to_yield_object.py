# A generator declared to yield `object` yields values of several types --
# an instance, None, an int, a str -- and the reader tells them apart with
# isinstance. Each value crosses the resume boxed; what the reader prints is
# what came out of the box.
from typing import Generator


class Fut:
    def __init__(self, n: int) -> None:
        self.n = n


def waits() -> Generator[object, None, int]:
    yield Fut(5)
    yield None
    yield 7
    yield "s"
    return 3


def drive(g: Generator[object, None, int]) -> None:
    for o in g:
        if isinstance(o, Fut):
            print("fut", o.n)
        elif isinstance(o, int):
            print("int", o)
        elif isinstance(o, str):
            print("str", o)
        else:
            print("other", o)


drive(waits())
