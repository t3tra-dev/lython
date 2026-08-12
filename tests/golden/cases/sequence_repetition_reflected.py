# Why execution: the values are what the fold has to preserve. `2 * "ab"` was
# "builtins.int does not provide manifest method '__mul__'" while `"ab" * 2`
# worked -- CPython gets to the same answer by returning NotImplemented from
# int.__mul__ and running the sequence's __rmul__, which for these four IS
# __mul__ with the operands swapped.
def main() -> None:
    print(2 * "ab", "ab" * 2)
    print(3 * [1, 2], [1, 2] * 3)
    print(2 * (1, 2), (1, 2) * 2)
    print(2 * b"ab", b"ab" * 2)
    print(0 * [1], [1] * 0, -1 * "x", "x" * -1)
    n = 3
    word = "xy"
    print(n * word, word * n)
    items = [0]
    print(n * items, items * n)
    # the operands still evaluate left to right, once each
    log: list[str] = []

    def left() -> int:
        log.append("left")
        return 2

    def right() -> str:
        log.append("right")
        return "z"

    print(left() * right(), log)


main()
