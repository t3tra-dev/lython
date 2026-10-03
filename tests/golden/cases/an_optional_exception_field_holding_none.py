# What: a field typed `BaseException | None` that holds None, read and
# compared. Reading the union took the exception member's message lanes from
# the empty member before the tag chose None, and followed a null message
# pointer: a segfault at run time, after everything compiled.
class Waiter:
    def __init__(self) -> None:
        self.exception: BaseException | None = None


class Child(Waiter):
    def __init__(self) -> None:
        super().__init__()


w = Waiter()
print(w.exception is None)
c = Child()
print(c.exception is None)
c.exception = ValueError("late")
print(c.exception is None, c.exception)
