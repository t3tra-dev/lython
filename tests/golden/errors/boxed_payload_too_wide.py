# A payload box carries at most five handles. A class whose physical
# expansion is wider used to be truncated silently at the box (the element
# read back lost its tail) and, being too wide, also dropped out of the
# boxed-method dispatch, so an existing __repr__ turned into a runtime abort.
# The width is known at the box, which is where it is now rejected.
class Q:
    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b

    def __repr__(self) -> str:
        return "Q"


print([Q(1, 2)])
