# `global X` for a name with no cell falls through to a LOCAL binding, so the
# assignment is a silent no-op and the module global keeps its old value. The
# declaration is an explicit statement that the assignment is not a local one,
# so binding a local is the one answer it cannot have.
#
# This was a `list[int]` until container globals got cells, then a
# `list[int] | None` until a union a function uses got one, and then a
# `type[X]` until class values carried their type object. What is left is a
# name the module binds more than once without an annotation: it stays
# value-bound, since a cell has one representation and the bindings promise
# none.
G = 0
G = 1


def f() -> None:
    global G
    G = 2


f()
print(G)
