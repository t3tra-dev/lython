# WHAT: a value whose handle is the same memref type as an `object` box -- a
# source-class instance, a list, a tuple, a range -- stored in an `object`
# module global and returned as an `object`, then printed.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the upcast took matching
# types for a matching representation and handed the value's own handle on
# as its box, so the program compiled and the repr hook read the value's
# words as a box's -- a SIGSEGV, or nothing printed. Only execution shows
# which object the slot names.


class A:
    def __init__(self, n: int) -> None:
        self.n = n

    def __repr__(self) -> str:
        return "A(" + str(self.n) + ")"


def make_instance() -> object:
    return A(3)


def make_list() -> object:
    return [1, 2]


def make_tuple() -> object:
    return (4, "x")


def make_range() -> object:
    return range(2, 5)


g: object = A(7)
print(g)
g = [5, 6]
print(g)
g = (7,)
print(g)
g = range(3)
print(g)
print(make_instance(), make_list(), make_tuple(), make_range())
