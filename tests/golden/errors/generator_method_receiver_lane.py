# A generator method is no longer inlined (inlining substituted the body's own
# result, so `b.each()` typed as None and the for-loop reported that None has no
# __iter__). It now routes through the bound function object like an async
# method, which reaches the real limit: the captured receiver is a user-class
# contract, and only builtins.int and manifest contracts with a rank-1 physical
# shape have a generator resume lane. The rejection must name that.
from typing import Iterator


class Box:
    def __init__(self, n: int) -> None:
        self._n: int = n

    def each(self) -> Iterator[int]:
        i: int = 0
        while i < self._n:
            yield i
            i = i + 1


b = Box(3)
for v in b.each():
    print(v)
