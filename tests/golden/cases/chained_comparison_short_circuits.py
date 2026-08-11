# Why execution: the assertion is what does NOT print. A chained comparison
# stops at the first false link -- `1 > 2 > s()` never calls `s` -- and every
# operand is evaluated exactly ONCE, so the middle of `1 < m() < 10` is one
# call, not two. The pairs used to be emitted eagerly, so the trailing
# operands ran; rewriting to `a op b and b op c` would have duplicated the
# middle instead, which is why it had not been done.

calls: int = 0


def side(value: int, label: str) -> int:
    global calls
    calls = calls + 1
    print("eval", label)
    return value


def reset() -> None:
    global calls
    calls = 0


def stops_at_the_first_false_link() -> None:
    reset()
    print(1 > 2 > side(0, "third"), calls)


def evaluates_the_middle_once() -> None:
    reset()
    print(1 < side(5, "middle") < 10, calls)


def stops_at_the_second_link() -> None:
    reset()
    print(1 < side(5, "middle") < 3, calls)
    reset()
    print(1 < side(5, "middle") < 3 < side(0, "fourth"), calls)


def four_operands_all_true() -> None:
    reset()
    print(0 < side(1, "b") < side(2, "c") < 3, calls)


def the_ordinary_range_check() -> None:
    lo = 48
    hi = 57
    print(lo <= 50 <= hi, lo <= 99 <= hi, lo <= 47 <= hi)


def mixed_operators() -> None:
    print(1 == 1 < 2, 3 > 2 == 2, 1 != 2 != 3, 1 < 2 > 1)


def main() -> None:
    stops_at_the_first_false_link()
    evaluates_the_middle_once()
    stops_at_the_second_link()
    four_operands_all_true()
    the_ordinary_range_check()
    mixed_operators()


main()
