# An annotation wider than the value assigned to it. Execution is needed
# because the defect was a RETYPING that compiled: the value carried an int's
# three lanes under a type that said float, and `print(x)` alone still printed
# 3, so only a use that reads the lanes tells the two apart. `x + 0.5` is that
# use -- it failed to lower before, and a repair that converted instead of
# leaving the value alone prints 3.5 here but 3.0 on the line above.
#
# CPython prints 3 for the first line: the annotation does not convert, and
# neither does this.


def widened() -> None:
    x: float = 3
    print(x)
    print(x + 0.5)
    print(x / 2)
    print(x * 2)
    print(str(x))
    print(x == 3.0)


widened()


def narrowed() -> None:
    # The bool rung of the same relation.
    n: int = True
    print(n, n + 1)


narrowed()


def exact() -> None:
    # No widening at all, so nothing above may have changed it.
    y: float = 2.5
    print(y, y + 0.5)


exact()
