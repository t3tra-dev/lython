# WHAT: a value of each kind stored where an `object` is declared -- a module
# global, a class field read back through a parameter, a closure's cell after
# a rebinding -- and read back, by print, str and repr.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: each of these compiled
# and read the wrong words at run time. The global's cell took the float's
# own handle where it keeps a box, and the field and cell reads took the
# payload the slot points at for the box around it; what showed was a SIGSEGV
# on the float's bits, or nothing printed at all.
g: object = 1.5
print(g, repr(g), str(g))
g = "two"
print(g, repr(g))
g = [3, 4.5]
print(g)
g = None
print(g)
g = (1, "b")
print(g)


def read_global() -> str:
    return repr(g)


g = {"k": 2.5}
print(read_global())


class Holder:
    o: object

    def __init__(self) -> None:
        self.o = 1.5


def show(h: Holder) -> str:
    return repr(h.o)


h = Holder()
print(show(h))
h.o = "s"
print(show(h))
h.o = [1, 2]
print(show(h))


def outer() -> None:
    c: object = 2.5

    def inner() -> None:
        print(repr(c))

    inner()
    c = "rebound"
    inner()
    c = [7.5]
    inner()


outer()
