# `global X` for a name with no cell falls through to a LOCAL binding, so the
# assignment is a silent no-op and the module global keeps its old value. The
# declaration is an explicit statement that the assignment is not a local one,
# so binding a local is the one answer it cannot have.
#
# This was a `list[int]` until container globals got cells, then a
# `list[int] | None` until a union a function uses got one, then a `type[X]`
# until class values carried their type object, and then a name bound twice
# without an annotation until such a name got a cell of its values' join.
# What is left is a name bound by unpacking: no single value types the cell.
G, H = 0, 0
G = 1


def f() -> None:
    global G
    G = 2


f()
print(G, H)
