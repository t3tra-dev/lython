# `global X` for a container global used to fall through to a LOCAL binding,
# so the assignment was a silent no-op and the module global kept its old
# value. A container is the one annotated shape with no cell to write -- its
# structural mutations reallocate the interior arrays through SSA rebinding,
# which a cell would go stale against -- and the declaration is an explicit
# statement that the assignment is not a local one, so binding a local is the
# one answer it cannot have.
X: list[int] = [1]


def f() -> None:
    global X
    X = [2]


f()
print(X)
