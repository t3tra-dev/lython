# WHAT: a range is true exactly when it has members, by `bool()`, by `if`, by
# `not` and by `and`/`or`, whatever its start, stop and step.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the length is the run
# time's. A range declares no `__bool__`, so its truth resolved to object's,
# which reads a box -- and the range's handle was handed over as one, so
# `bool(range(3))` crashed.


def make(start: int, stop: int, step: int) -> range:
    return range(start, stop, step)


for r in [range(3), range(0), range(5, 5), range(0, 10, 3), range(3, 0, -1),
          range(0, 3, -1), make(2, 9, 4), make(9, 2, 1)]:
    print(r, bool(r), "y" if r else "n", not r, len(r))
print(bool(range(1)) and not range(0), range(0) or "empty")
