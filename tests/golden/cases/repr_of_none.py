# What: `repr(None)`. The repr builtin had its own copy of the render ladder,
# and a None resolves a manifest `__repr__` there -- so it emitted a py.repr on
# a value that expands to NO physical values and died in the LOWERING as
# "types.NoneType runtime object has no physical header value", for a two-word
# program. `f"{v!r}"`, `str(v)` and `print(v)` all render it, all three through
# the shared ladder; this was the one door with its own copy.
#
# Running it is the point: what the ladder produces is the STRING "None", and
# every other repr on the same line has to keep producing what it did.
from typing import Optional


class Box:
    def __init__(self, n: int) -> None:
        self.n = n

    def __repr__(self) -> str:
        return "Box(" + str(self.n) + ")"


def maybe(flag: bool) -> Optional[int]:
    return 3 if flag else None


empty: None = None

print("direct", repr(None))
print("through a local", repr(empty))
print("through an optional", repr(maybe(True)), repr(maybe(False)))
print("inside containers", repr([None, None]), repr((None, 1)), repr({"a": None}))
print("the others still", repr(Box(2)), repr("s"), repr(3), repr(True), repr(1.5))
print("conversions agree", f"{empty!r}", str(empty), f"{empty}")
