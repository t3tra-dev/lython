# probe: REPORTED loud: a class field holding a callable
# axes: width=callable op=field flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   static type !py.contract<"Box"> does not provide manifest method 'fn'
# CPython 3.14 expects: 42

from typing import Callable


def double(n: int) -> int:
    return n * 2


class Box:
    def __init__(self, fn: Callable[[int], int]) -> None:
        self.fn: Callable[[int], int] = fn


o = Box(double)
print(o.fn(21))
