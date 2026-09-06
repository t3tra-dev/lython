# What: a nested callable that captures an `Optional[T]` and OUTLIVES the frame
# that made it. A capture is stored as one box, and reading it back asked for
# the VALUE's own lanes -- a union's first lane is its i64 TAG, so the rebuild
# said " has no statically sized entity lane to rebuild a box from, got 'i64'".
# A union has no contract name to find a `box` primitive under, which is why
# the shape that answers for every other capture answered nothing here.
#
# Running it is the evidence the tag survived the box: the closure is called
# after its maker returned, so what it reads comes out of the store rather than
# out of the frame, and both arms of the guard have to be reachable.
from typing import Callable, Optional


def held(seed: Optional[int]) -> Callable[[], int]:
    v = seed

    def inner() -> int:
        if v is None:
            return -1
        return v + 1

    return inner


def held_str(seed: Optional[str]) -> Callable[[], int]:
    v = seed

    def inner() -> int:
        return 0 if v is None else len(v)

    return inner


def rebound(seed: Optional[int]) -> Callable[[], int]:
    v = seed

    def inner() -> int:
        return 0 if v is None else v

    v = 41
    return inner


print("value", held(3)(), held(None)())
print("string", held_str("abcd")(), held_str(None)())
print("rebound", rebound(None)(), rebound(7)())
