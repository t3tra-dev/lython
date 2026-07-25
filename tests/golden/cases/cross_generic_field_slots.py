# Cross-track: Wave 3's generic-class monomorphization (`C$specN`) meets
# kernel/4a's field slots. A specialization's field storage is chosen from the
# SUBSTITUTED field type, so `self.v: T` and `self.items: dict[int, T]` have to
# reach the same box16 slot path as a concrete annotation, and a field whose
# declared type is itself a specialization (`Stack[int]`, `OrderedDict[str,
# int]`, `Cell[int]`) has to be a slot holding one handle rather than the
# specialization's inlined expansion.
#
# Both halves of 4a are exercised on those fields: mutation of the pointee
# through a borrowed parameter (`fill`, `bump`), and REBIND of the field itself
# through a borrowed parameter (`rebind`, `swap_stack`), which is the store that
# used to compile to a callee writing nothing the caller could observe.
from collections import OrderedDict


class Cell[T]:
    def __init__(self, v: T) -> None:
        self.v: T = v

    def get(self) -> T:
        return self.v

    def set(self, v: T) -> None:
        self.v = v


class Stack[T]:
    def __init__(self) -> None:
        self.items: dict[int, T] = {}
        self.size = 0

    def push(self, item: T) -> None:
        self.items[self.size] = item
        self.size = self.size + 1

    def pop(self) -> T:
        self.size = self.size - 1
        item = self.items[self.size]
        del self.items[self.size]
        return item

    def to_list(self) -> list[T]:
        out: list[T] = []
        index = 0
        while index < self.size:
            out.append(self.items[index])
            index = index + 1
        return out


class Holder:
    def __init__(self, s: Stack[int], o: OrderedDict[str, int]) -> None:
        self.s: Stack[int] = s
        self.o: OrderedDict[str, int] = o


def fill(h: Holder) -> None:
    h.s.push(1)
    h.s.push(2)
    h.o["a"] = 1
    h.o["b"] = 2


def mk() -> Holder:
    fresh: Stack[int] = Stack()
    table: OrderedDict[str, int] = OrderedDict()
    return Holder(fresh, table)


h = mk()
fill(h)
print(h.s.to_list(), len(h.o))
print(h.s.pop(), h.s.to_list())
print(list(h.o.keys()), list(h.o.values()))
print(h.o["a"], h.o["b"], "a" in h.o)

ci: Cell[int] = Cell(3)
cs: Cell[str] = Cell("ab")
print(ci.get(), cs.get())


def bump(c: Cell[int]) -> None:
    c.set(c.get() + 10)


bump(ci)
print(ci.get(), ci.v)


class Nest:
    def __init__(self, c: Cell[int]) -> None:
        self.c: Cell[int] = c


nest = Nest(Cell(5))
print(nest.c.v, nest.c.get())
nest.c.set(6)
print(nest.c.v)


def rebind(n: Nest) -> None:
    n.c = Cell(99)


rebind(nest)
print(nest.c.v, nest.c.get())


def swap_stack(hold: Holder) -> None:
    other: Stack[int] = Stack()
    other.push(7)
    hold.s = other


swap_stack(h)
print(h.s.to_list(), len(h.o))
