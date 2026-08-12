# Why execution: these were refused, and the refusal named the wrong thing --
# "unresolved name 'max'" for a builtin whose two-argument form works. What the
# golden pins is the VALUE each form produces, since a fold that computed the
# wrong association or seeded the wrong accumulator would still compile.


def main() -> None:
    print(max(1, 2, 3))
    print(min(3, 1, 2))
    print(max(4, 2, 3, 1))
    print(min("b", "a", "c", "d"))

    print(max(1, 2))
    print(min(2, 1))

    numbers: list[int] = [1, 2, 3]
    print(sum(numbers, 10))
    print(sum(numbers, -10))
    print(sum(numbers))
    print(max(numbers))
    print(min(numbers))


main()


# The accumulator is one SSA value with one type, so an implicit int 0 seed
# asked the lowering to store a float into an int lane -- "cannot adapt
# runtime bundle builtins.float with physical values (memref<3xi64>) to
# expected ABI". CPython's seed is the int 0 and `0 + 1.5` promotes; seeding
# the promoted zero is the same answer, and the int seed stays wherever the
# promotion does not happen -- including the empty iterable.
def floats() -> None:
    values: list[float] = [1.5, 2.5]
    print(sum(values))
    print(sum(values) + 1)
    print(sum(values, 10.0))
    print(sum(v * 0.5 for v in [1, 2]))
    empty: list[int] = []
    print(sum(empty))


floats()
