# WHAT: a generator whose frame carries a BOOL across a yield, drained in a
# loop. A bool lane is one word and marks no ownership -- its DecRef is a no-op
# -- so the frame's aggregate release must not double-count it; the string
# lane beside it is what the release really has to cover, sized past the probe
# floor.
from typing import Iterator


def pairs(tag: str) -> Iterator[str]:
    flag = True
    for _ in range(2):
        yield tag if flag else ""
        flag = not flag


i = 0
while i < 3000:
    for value in pairs("z" * 4096):
        if len(value) > 0:
            pass
    i += 1
print("done")
