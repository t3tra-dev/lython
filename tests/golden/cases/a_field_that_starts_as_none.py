# `self.x = None` in the constructor and a value later is how Python declares a
# slot that starts empty. The field type was whatever the FIRST assignment
# said, so the slot stayed NoneType and every later store died in the lowering
# as "attribute value 'builtins.int' is not assignable to field
# '!py.literal<None>'" -- while `self.x: int | None = None` has always worked.


class Node:
    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt = None

    def link(self, other: "Node") -> None:
        self.nxt = other


head = Node(1)
head.link(Node(2))
tail = head.nxt
print(head.v, tail.v if tail is not None else -1)


class Config:
    def __init__(self) -> None:
        self.name = None
        self.size = None
        self.tags = None

    def load(self, name: str, size: int) -> None:
        self.name = name
        self.size = size
        self.tags = [name]

    def show(self) -> str:
        n = self.name
        s = self.size
        if n is not None and s is not None:
            return n + ":" + str(s)
        return "empty"


c = Config()
print(c.show(), c.tags)
c.load("a", 3)
print(c.show(), c.tags)


# A method may also give the field its value through ANOTHER instance of the
# same class -- the shape every linked structure is written in.
class Cell:
    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt = None

    def chain(self, other: "Cell") -> None:
        other.nxt = self


first = Cell(1)
second = Cell(2)
first.chain(second)
print(second.nxt.v if second.nxt is not None else -1, first.nxt)


# The other order too: a value first, cleared to None later.
class Slot:
    def __init__(self) -> None:
        self.n = 1

    def clear(self) -> None:
        self.n = None


s = Slot()
print(s.n)
s.clear()
print(s.n)
