# What: a nested function that reads a name the enclosing body binds LATER --
# mutually recursive defs, a one-way forward reference, and a plain value.
# CPython gives every closed-over name a cell, so the read happens when the
# inner function RUNS. Runtime values, because the question is which value each
# inner call sees, and the same three shapes at module scope already worked.


def parity(n: int) -> str:
    def is_even(k: int) -> bool:
        if k == 0:
            return True
        return is_odd(k - 1)

    def is_odd(k: int) -> bool:
        if k == 0:
            return False
        return is_even(k - 1)

    return "even" if is_even(n) else "odd"


def one_way(n: int) -> int:
    def caller(k: int) -> int:
        return helper(k) * 2

    def helper(k: int) -> int:
        return k + 1

    return caller(n)


def value_after(n: int) -> int:
    def scaled(k: int) -> int:
        return k * factor

    factor = 10
    return scaled(n)


def rebound_after(n: int) -> int:
    def read() -> int:
        return base

    base = 1
    first = read()
    base = 7
    return first * 100 + read() + n


class Walk:
    def run(self, n: int) -> int:
        def down(k: int) -> int:
            if k == 0:
                return 0
            return up(k - 1) + 1

        def up(k: int) -> int:
            return down(k)

        return down(n)


print(parity(6), parity(7))
print(one_way(3))
print(value_after(4))
print(rebound_after(5))
print(Walk().run(4))
