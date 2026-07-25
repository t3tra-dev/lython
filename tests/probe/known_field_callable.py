# probe: REPORTED loud: a class field holding a callable
# axes: width=callable op=field flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   /Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe/known_field_callable.py:19:6: emit error: static type !py.contract<"Box"> does not provide manifest method 'fn'
# CPython 3.14 expects: 42

from typing import Callable


def double(n: int) -> int:
    return n * 2


class Box:
    def __init__(self, fn: Callable[[int], int]) -> None:
        self.fn: Callable[[int], int] = fn


o = Box(double)
print(o.fn(21))
