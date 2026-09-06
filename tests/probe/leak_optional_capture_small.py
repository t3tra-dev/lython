# WHAT: a closure capturing an Optional[str], built and called in a loop. The
# capture is one box and the read rebuilds the union's lanes from it, so
# whatever the rebuild hands back must not be double-counted against the
# store's own reference. The string is sized past the probe floor.
from typing import Callable, Optional


def held(seed: Optional[str]) -> Callable[[], int]:
    v = seed

    def inner() -> int:
        return 0 if v is None else len(v)

    return inner


i = 0
while i < 300:
    f = held("z" * 4096)
    if f() != 4096:
        print("never")
    i += 1
print("done")
