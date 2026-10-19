# WHAT: `f.__name__` through a parameter that holds a function: it is folded
#   at compile time from the def. assertRaises' callable form names the
#   callable in its failure message this way.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   '__name__' of a function is folded at compile time and needs the name the `def` gave; write it through that name (or through the class for a method) rather than through a value that holds the function
# CPython 3.14 expects: hello
from collections.abc import Callable


def describe(f: Callable[[], None]) -> str:
    return f.__name__


def hello() -> None:
    pass


print(describe(hello))
