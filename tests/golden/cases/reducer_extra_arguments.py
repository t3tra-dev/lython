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


# key= rides the same loop as one more carried accumulator, the way CPython's
# builtin_max does it: the key function is applied once per item and the ITEM
# is what is kept. Ties keep the first, which is the strict comparison.
def keyed() -> None:
    words = ["bb", "a", "ccc"]
    print(max(words, key=len), min(words, key=len))
    print(max(["ab", "cd"], key=len), min(["ab", "cd"], key=len))

    def neg(v: int) -> int:
        return -v

    print(max([3, 1, 2], key=neg), min([3, 1, 2], key=neg))
    try:
        print(max([], key=len))
    except ValueError as err:
        print(err)


keyed()
