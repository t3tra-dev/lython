# An exception raised in a `with` body and handled in the generator of a
# @contextmanager carries the frames it was raised through: CPython keeps them
# on the exception, and `__exit__` throwing it into the generator puts the
# generator's frame in front of them. The handler saw the generator's frame
# alone, so unittest's failure reports -- written from such a handler --
# showed no frame of the test. One the generator does not handle leaves with
# the frames it came with, and the next exception starts with none of them.
# Why run it: the frames are what the handler reads at run time.
import contextlib
import sys
import traceback
from collections.abc import Iterator


def where() -> list[str]:
    return [entry.split(", line ")[1].split("\n")[0]
            for entry in traceback.format_tb(sys.exc_info()[2])]


@contextlib.contextmanager
def catch() -> Iterator[None]:
    try:
        yield
    except ValueError:
        print("handled:", where())


def inner() -> None:
    raise ValueError("x")


def outer() -> None:
    inner()


def runs_a_body() -> None:
    with catch():
        outer()


with catch():
    outer()
runs_a_body()
try:
    with catch():
        raise KeyError("not handled there")
except KeyError:
    print("escaped:", where())
try:
    inner()
except ValueError:
    print("next:", where())
