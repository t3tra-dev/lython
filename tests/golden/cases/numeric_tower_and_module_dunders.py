# Why execution: none of these compiled. Mixed int/float arithmetic was
# refused outright ("builtins.int does not provide manifest method '__add__'"),
# `if __name__ == "__main__":` was "unresolved name '__name__'", and `if s:` on
# a set was "runtime manifest has no builtins.set.__bool__ method". The values
# are the assertion because the promotion has to produce CPython's numbers,
# not just compile.


def mixed_arithmetic() -> None:
    a: int = 1
    b: float = 2.5
    print(a + b, b + a, a - b, b - a)
    print(a * b, b * a, a / b, b / a)
    print(a // b, b // a, a % b, b % a)
    print(1 + 2.0, 2.0 + 1, 7 // 2.0, 7.0 // 2)


def mixed_comparison() -> None:
    a: int = 1
    b: float = 2.5
    print(a < b, b < a, a == 1.0, 1.0 == a, a <= 1.0, b >= 2.5)


def int_division_stays_exact() -> None:
    # CPython scales the two integers instead of converting each to a double,
    # so this is 10.0 and not an OverflowError.
    print(10**400 / 10**399)
    print(1 / 2, 7 / 2, -7 / 2)


def set_truthiness() -> None:
    s: set[int] = {1}
    e: set[int] = set()
    if s:
        print("non-empty")
    if not e:
        print("empty")
    print(bool(s), bool(e))


def main() -> None:
    mixed_arithmetic()
    mixed_comparison()
    int_division_stays_exact()
    set_truthiness()


# `__name__` is a module-level binding, and module bindings are not visible
# from a function body yet -- so it is read here, where the idiom lives.
print(__name__)
if __name__ == "__main__":
    main()
