# A `Callable[[], object]` value is called, and one function the dispatch
# must consider takes `*names`. Building its empty `*args` tuple moved the
# builder to the call itself, so that candidate's arm was left without even
# its branch ("empty block: expect at least a terminator") -- for any program
# that calls a callable taken from a list while a variadic function exists.
# Why run it: each arm has to call its own target with its own arguments.
from collections.abc import Callable


def report(*names: str) -> None:
    print("report", names, len(names))


def hello() -> object:
    return "hello"


calls: list[Callable[[], object]] = [hello, report]
for c in calls:
    print(c())
