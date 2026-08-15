# `global X` for a name with no cell falls through to a LOCAL binding, so the
# assignment is a silent no-op and the module global keeps its old value. The
# declaration is an explicit statement that the assignment is not a local one,
# so binding a local is the one answer it cannot have.
#
# This was a `list[int]` until container globals got cells. A union is what is
# left: it stays value-bound so isinstance narrowing keeps working on the
# module flow, which means there is nothing for the write to reach.
X: list[int] | None = [1]


def f() -> None:
    global X
    X = [2]


f()
print(X)
