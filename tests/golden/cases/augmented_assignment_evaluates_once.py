# Why execution: the COUNT is the assertion. `a[f()] += 1` rewrote to a load
# and a store of the same target subtree, so `f()` ran twice and the store
# landed wherever the SECOND call pointed -- [0, 0, 1] where CPython gives
# [0, 1, 0]. Only running it shows either the count or the placement.

calls: int = 0


def bump() -> int:
    global calls
    calls = calls + 1
    return calls


class Cell:
    def __init__(self) -> None:
        self.v = 0


def receiver(c: Cell) -> Cell:
    global calls
    calls = calls + 1
    return c


def reset() -> None:
    global calls
    calls = 0


def index_expression() -> None:
    reset()
    a: list[int] = [0, 0, 0]
    a[bump()] += 1
    print(a, calls)


def receiver_expression() -> None:
    reset()
    c = Cell()
    receiver(c).v += 1
    print(c.v, calls)


def slice_bounds() -> None:
    reset()
    a: list[int] = [1, 2, 3, 4, 5]
    a[bump():3] += [99]
    print(a, calls)


def in_place_container_method() -> None:
    reset()
    g: list[list[int]] = [[1], [2]]
    g[bump()] += [7]
    print(g, calls)


def a_plain_name_needs_nothing() -> None:
    reset()
    n = 1
    n += 2
    n *= 3
    print(n, calls)


def a_constant_index() -> None:
    reset()
    a: list[int] = [0, 0]
    a[1] += 5
    print(a, calls)


def main() -> None:
    index_expression()
    receiver_expression()
    slice_bounds()
    in_place_container_method()
    a_plain_name_needs_nothing()
    a_constant_index()


main()
