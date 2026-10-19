# WHAT: A generic Protocol (`class SupportsLT[T](Protocol)`); typeshed's
#   SupportsDunderLT is one.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   a generic Protocol (`class SupportsLT[T](Protocol)`) is not supported yet: a class that satisfies it would need its type arguments inferred from its members
# CPython 3.14 expects: 1
from typing import Protocol


class SupportsLT[T](Protocol):
    def __lt__(self, other: T, /) -> bool: ...


def smaller[T: SupportsLT[int]](a: T, b: T) -> T:
    return a if a < b else b


print(smaller(1, 2))
