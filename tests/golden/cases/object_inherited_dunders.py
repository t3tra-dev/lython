# What: builtins.object's defaults reach a user class through its MRO.
#   __eq__/__ne__ are identity, __hash__ is identity-derived, __bool__ is
#   always true (unless __len__ answers), and __repr__/__str__ are the
#   address form naming the DEFINING class -- while any definition earlier in
#   the linearization still wins, including the __ne__ CPython derives from a
#   lone __eq__.
# Addresses are not pinned (they vary per run); the class name in the repr and
# the repr/str agreement are, because those are what a fall-through to
# object's own manifest __repr__ would get wrong.


class Plain:
    def __init__(self, tag: int) -> None:
        self.tag = tag


class Derived(Plain):
    pass


class WithEq:
    def __init__(self, tag: int) -> None:
        self.tag = tag

    def __eq__(self, other: "WithEq") -> bool:
        return self.tag == other.tag


class Empty:
    def __len__(self) -> int:
        return 0


a = Plain(1)
b = Plain(1)
# Equal fields, distinct objects: the default compares identity, not state.
print(a == a, a == b, a != a, a != b)
print(hash(a) == hash(a))
print(bool(a), not a)
# object.__str__ IS type(x).__repr__, so the two agree.
print(repr(a) == str(a))
print(repr(a).startswith("<__main__.Plain object at 0x"))
print(a.__repr__() == a.__str__())
# A base that defines none of them passes the defaults through.
d = Derived(2)
print(d == d, d != d, bool(d))
print(repr(d).startswith("<__main__.Derived object at 0x"))
# A user __eq__ outranks the default, and != negates THAT __eq__.
p = WithEq(3)
q = WithEq(3)
print(p == q, p != q)
# __len__ answers truthiness before object's default does.
print(bool(Empty()))
