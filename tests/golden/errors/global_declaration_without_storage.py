# `global X` for a name with no cell falls through to a LOCAL binding, so the
# assignment is a silent no-op and the module global keeps its old value. The
# declaration is an explicit statement that the assignment is not a local one,
# so binding a local is the one answer it cannot have.
#
# This was a `list[int]` until container globals got cells, and then a
# `list[int] | None` until a union a function uses got one. A class object is
# what is left: `type[X]` is compile-time evidence with no runtime value group,
# so there is nothing for the write to reach.
class A:
    pass


class B(A):
    pass


X: type[A] = A


def f() -> None:
    global X
    X = B


f()
print(X.__name__)
