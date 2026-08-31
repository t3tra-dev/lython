# What: a field that holds a lambda has to be callable through the instance,
# and the value the call returns is the only thing that shows the stored
# function is the one the constructor put there.
class Box:
    def __init__(self, n: int) -> None:
        self.n = n
        self.get = lambda: 6
        self.tag = lambda: "boxed"


b = Box(2)
print(b.n, b.get(), b.tag())


class Pair:
    def __init__(self) -> None:
        self.first = lambda: 1
        self.second = lambda: 2


p = Pair()
print(p.first(), p.second())


def seven() -> int:
    return 7


class Mixed:
    def __init__(self) -> None:
        self.named = seven
        self.anon = lambda: 8


m = Mixed()
print(m.named(), m.anon())
