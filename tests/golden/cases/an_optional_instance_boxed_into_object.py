# A `Tb | None` of a program class boxed into `object` -- returned from a
# function declared to return object, and `sys.exc_info()` in a handler that
# a callable taken from a list raised into. Boxing reads each member's class
# word and lets the tag pick; the verifier refused that read as a use of the
# payload before its tag was tested, while the same union of a builtin (whose
# class word is a constant) was accepted.
# Why run it: the box must name the member that is live, which only the
# printed answer shows.
import sys
from collections.abc import Callable


class Tb:
    def __init__(self, n: int) -> None:
        self.n = n


def maybe(flag: bool) -> Tb | None:
    if flag:
        return Tb(1)
    return None


def as_object(flag: bool) -> object:
    return maybe(flag)


print(as_object(True) is None, as_object(False) is None)
print(type(as_object(True)).__name__, type(as_object(False)).__name__)


def boom() -> object:
    raise KeyError("k")


cleanups: list[Callable[[], object]] = [boom]
try:
    cleanups[0]()
except Exception:
    exc_info = sys.exc_info()
    print("caught", type(exc_info[1]).__name__, exc_info[1])
print("done")
