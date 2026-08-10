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
