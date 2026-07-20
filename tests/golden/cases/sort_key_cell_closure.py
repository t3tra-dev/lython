# w15 carryover repro: a key lambda over an R6 shared cell with a 6+ element
# inline literal used to lose the list bundle ("no builtins.list.__getitem__
# method"); pinned here against CPython.


def outer() -> None:
    calls = 0

    def bump() -> None:
        nonlocal calls
        calls = calls + 1

    bump()
    print(sorted([5, 2, 8, 1, 9, 3], key=lambda x: x + calls))
    print(sorted([5, 2, 8, 1, 9], key=lambda x: x + calls))
    print(sorted([5, 2, 8, 1, 9, 3, 7, 6], key=lambda x: -(x + calls)))


outer()
