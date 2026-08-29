# Two closures over the SAME captured object, returned together. Refused with
# "ownership: this block-argument merge needs a retain on the edge and the
# header prefix cannot be spelled at the point the retain must go (header type
# 'memref<9xi64>', op result)".
#
# The header was a heap DEAD PLACEHOLDER -- the value a merge takes on the edge
# where the object does not exist. It is allocated raw and its refcount word is
# written afterwards, so nothing at the `memref.alloc` proved the word was
# there, and the refusal was right to protect a retain from reading it. What
# was missing is the marker that says the entity is complete FROM A POINT.
#
from typing import Callable


# Golden because the placeholder is a real object with a refcount: marking it
# moves where the retain goes, and a wrong move is an over- or under-release
# rather than a compile error. Registered in the leak gate for the same reason.
def make_pair() -> "tuple[Callable[[int], int], Callable[[], list[int]]]":
    values: list[int] = []

    def add(x: int) -> int:
        values.append(x)
        return len(values)

    def read() -> list[int]:
        return values

    return add, read


add, read = make_pair()
print(read())
print(add(1), add(2), read())


def counters() -> "tuple[Callable[[str], int], Callable[[], int]]":
    seen: dict[str, int] = {}

    def bump(key: str) -> int:
        seen[key] = seen.get(key, 0) + 1
        return seen[key]

    def total() -> int:
        out = 0
        for value in seen.values():
            out += value
        return out

    return bump, total


bump, total = counters()
for key in ["a", "b", "a"]:
    bump(key)
print(total())

for _ in range(3):
    add, read = make_pair()
    add(9)
    print(read())
