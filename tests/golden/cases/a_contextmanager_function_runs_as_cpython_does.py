# WHAT: `@contextlib.contextmanager` (both spellings of the import) on a
# function and on a method: the value `yield` hands to `as`, the body's normal
# end, an exception the generator catches and one it lets through, the
# `else`/`finally` arms around the yield, and a generator that yields twice.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which arm of the
# generator runs, and whether the exception comes back out, is decided at run
# time by how the generator ends after the exception is thrown into it. The
# decorator was refused ("decorator 'contextlib.contextmanager' is not
# supported"); unittest's own `_Outcome.testPartExecutor` is one.
import contextlib
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def tag(name: str) -> Iterator[int]:
    print("enter", name)
    try:
        yield len(name)
    finally:
        print("exit", name)


with tag("abc") as n:
    print("body", n)


class Case:
    def __init__(self) -> None:
        self.log: list[str] = []

    @contextlib.contextmanager
    def part(self, label: str) -> Iterator[str]:
        self.log.append("enter " + label)
        try:
            yield label.upper()
        except AssertionError as e:
            self.log.append("failed " + label + ": " + str(e))
        else:
            self.log.append("ok " + label)
        finally:
            self.log.append("exit " + label)


c = Case()
with c.part("a") as up:
    c.log.append("body " + up)
with c.part("b"):
    assert False, "nope"
try:
    with c.part("c"):
        raise KeyError("k")
except KeyError as e:
    c.log.append("propagated " + repr(e))
print(c.log)

log: list[str] = []


@contextlib.contextmanager
def outcome():
    try:
        yield
    except ValueError as e:
        log.append("error " + str(e))
    else:
        log.append("ok")


with outcome():
    pass
with outcome():
    raise ValueError("boom")
print(log)


@contextmanager
def twice() -> Iterator[int]:
    yield 1
    yield 2


try:
    with twice() as v:
        print("v", v)
except RuntimeError as e:
    print("RuntimeError:", e)
