# WHAT: `choice: A | None = None`, then `choice = B(...)` (B a subclass of A)
# inside an `if`, then `if choice is not None: choice.h` after it.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: it compiled; the read
# after the `if` dereferenced an empty lane and the program died with SIGSEGV.
#
# ⛔ The two arms were joined to `A | B | None`, and the read unwrapped the A
# member while the B member was the one present. argparse's
# `add_parser(..., help=...)` is this shape.


class A:
    def __init__(self, h: str) -> None:
        self.h = h


class B(A):
    pass


def pick(flag: bool) -> str:
    choice: A | None = None
    if flag:
        choice = B("from B")
    if choice is not None:
        return choice.h
    return "none"


print(pick(True))
print(pick(False))
