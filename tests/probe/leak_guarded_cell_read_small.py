# WHAT: a guard over a CELL-backed optional, read in a loop. The read now
# tests the tag and UNWRAPS the payload out of the cell's box on every trip, so
# whatever the unwrap hands back has to be released -- the string is sized past
# the probe floor and the cell is rebound each iteration.
from typing import Optional


def make(seed: Optional[str]) -> int:
    v = seed

    def inner() -> int:
        if v is None:
            return 0
        return len(v)

    v = "z" * 4096
    return inner()


i = 0
while i < 300:
    if make(None) != 4096:
        print("never")
    i += 1
print("done")
