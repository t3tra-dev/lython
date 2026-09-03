# `isinstance(a, M)` where nothing connects A and M in the type system, but a
# third class derives from both. Answering False there is silent: the branch
# the program wrote simply does not run, and the value it falls through to is a
# plausible one.


class Node:
    def __init__(self, n: int) -> None:
        self.n = n


class Serializable:
    def dump(self) -> str:
        return "{}"


class Leaf(Node, Serializable):
    def dump(self) -> str:
        return "leaf" + str(self.n)


class Twig(Node, Serializable):
    pass


def emit(n: Node) -> str:
    if isinstance(n, Serializable):
        return "yes"
    return "no"


print(emit(Node(1)), emit(Leaf(2)), emit(Twig(3)))


# With ONE class deriving from both, the true branch narrows and the mixin's
# method is reachable through the narrowed value.
class Plain:
    pass


class Tagged:
    def tag(self) -> str:
        return "t"


class Both(Plain, Tagged):
    def tag(self) -> str:
        return "both"


def describe(p: Plain) -> str:
    if isinstance(p, Tagged):
        return p.tag()
    return "-"


print(describe(Plain()), describe(Both()))


# And a class that shares no subclass with the target still folds to False.
class Alone:
    pass


def never(p: Plain) -> bool:
    return isinstance(p, Alone)


print(never(Plain()), never(Both()))
