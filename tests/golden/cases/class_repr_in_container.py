# A user class inside a container renders through the boxed __repr__ hook,
# which reconstructs the receiver from the slot box's payload handles. A str
# field occupies one box-fronted handle (not the payload's two lanes), so a
# (str, int) value class fits the box and reaches its own __repr__ instead of
# falling out of the hook's dispatch and aborting at runtime.
class P:
    def __init__(self, n: str, v: int) -> None:
        self.n = n
        self.v = v

    def __repr__(self) -> str:
        return "P(" + self.n + "," + repr(self.v) + ")"


print([P("a", 1), P("b", 2)])
print(repr(P("c", 3)))
print([P("d", 4)])
