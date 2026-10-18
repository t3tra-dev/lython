# WHAT: `dir()` of a class, of a class value that may name a subclass, and of
# an instance lists what CPython 3.14 lists: object's names, the five a class
# statement adds, `__annotate_func__` once a body in the line annotates, the
# class attributes of the line (an annotation without a value is not one),
# and for an instance the fields stored on it -- one a later method stores
# only once it has.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the class a value names
# and the fields an instance holds are run-time facts. `dir` was an
# unresolved name.
class A:
    x: int = 1
    y: str

    def __init__(self) -> None:
        self.f = 2

    def m(self) -> None:
        pass

    def later(self) -> None:
        self.w = 3


class B(A):
    def test_b(self) -> None:
        pass

    def test_a(self) -> None:
        pass

    def helper(self) -> None:
        pass


class C:
    """doc"""

    def go(self) -> None:
        pass


print(dir(A))
print([n for n in dir(B) if n.startswith("test")])
print(dir(C) == sorted(dir(C)), len(dir(C)), "__annotate_func__" in dir(C))


def names(cls: type[A]) -> list[str]:
    return [n for n in dir(cls) if not n.startswith("__")]


print(names(A), names(B))
a = A()
print([n for n in dir(a) if not n.startswith("__")])
a.later()
print([n for n in dir(a) if not n.startswith("__")])


def inst(o: A) -> list[str]:
    return [n for n in dir(o) if not n.startswith("__")]


b = B()
print(inst(b))
b.later()
print(inst(b))
