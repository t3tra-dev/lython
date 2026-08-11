# Why execution: these did not compile -- "borrowed entry argument 0 of @clamp
# is returned as owned without a dominating retain". The retain that was
# missing is a real reference, so the golden pins the returned VALUES and is
# in the leak gate: a fix that retained on the wrong path would compile and
# then leak or double-free.
#
# Two candidates fold to an arith.select and were always fine; three or more
# join through a block argument, which the retain-insertion walk did not
# follow.


def clamp_expression(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else (hi if x > hi else x)


def clamp_statements(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def pick_two(a: int, b: int) -> int:
    return a if a < b else b


def pick_strings(a: str, b: str, c: str) -> str:
    if a < b:
        return a
    if b < c:
        return b
    return c


def four_ways(a: int, b: int, c: int, d: int) -> int:
    if a > 0:
        return a
    if b > 0:
        return b
    if c > 0:
        return c
    return d


def main() -> None:
    print(clamp_expression(5, 0, 10))
    print(clamp_expression(-3, 0, 10))
    print(clamp_expression(42, 0, 10))
    print(clamp_statements(5, 0, 10))
    print(clamp_statements(-3, 0, 10))
    print(clamp_statements(42, 0, 10))
    print(pick_two(1, 2))
    print(pick_strings("m", "b", "z"))
    print(four_ways(0, 0, 3, 4))


main()
