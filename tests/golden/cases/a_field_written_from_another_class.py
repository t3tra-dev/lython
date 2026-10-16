# Why execution: the values read back through the widened fields are the
# subject -- a field that stayed NoneType would read None, and a subclass
# stored through a base-typed field must still be itself.
#
# An unannotated field seeded `None` in its own constructor and given a value
# only from ANOTHER class's method, through a field path (`self.tail.nxt =
# node` inside Queue names Cell.nxt), stayed NoneType: "attribute value 'Cell'
# is not assignable to field '!py.literal<None>'". The other class may be
# declared after, as Queue is after Cell, so the store is recorded with its
# path and applied once every class is declared.
#
# And a subclass instance stored into an Optional-of-base field through an
# inlined method (`adopt(SubLeaf(...))`) was refused in the lowering: the
# union's member was looked up without the class hierarchy.
class Cell:
    def __init__(self, v: int) -> None:
        self.v: int = v
        self.nxt = None


class Queue:
    def __init__(self) -> None:
        self.head = None
        self.tail = None

    def push(self, v: int) -> None:
        node = Cell(v)
        if self.tail is None:
            self.head = node
            self.tail = node
        else:
            self.tail.nxt = node
            self.tail = node

    def drain(self) -> list[int]:
        out: list[int] = []
        cur = self.head
        while cur is not None:
            out.append(cur.v)
            cur = cur.nxt
        return out


q = Queue()
for i in [1, 2, 3]:
    q.push(i)
print(q.drain())


class Leaf:
    def __init__(self, v: int) -> None:
        self.v = v
        self.label = None


class SubLeaf(Leaf):
    def kind(self) -> str:
        return "sub"


class Node:
    def __init__(self) -> None:
        self.kids: list[Leaf] = []
        self.first = None

    def adopt(self, leaf: Leaf) -> None:
        leaf.label = "kid" + str(leaf.v)
        self.kids.append(leaf)
        if self.first is None:
            self.first = leaf
        else:
            self.first.label = "first-of-" + str(len(self.kids))


class Tree:
    def __init__(self) -> None:
        self.root = Node()

    def relabel(self, text: str) -> None:
        root = self.root
        if root.first is not None:
            root.first.label = text


t = Tree()
t.root.adopt(SubLeaf(9))
for i in [1, 2]:
    t.root.adopt(Leaf(i))
first = t.root.first
print(first.label if first is not None else "-")
t.relabel("top")
for k in t.root.kids:
    print(k.v, k.label, isinstance(k, SubLeaf))
print(Leaf(5).label)
