# Why execution: every one of these COMPILED and printed the wrong value, so
# only the values show it. Three unrelated-looking defects that are all "the
# rebind went somewhere the program cannot see".


class M:
    def __init__(self, v: int) -> None:
        self.v = v

    def __add__(self, other: "M") -> "M":
        return M(self.v + other.v + 100)

    def __iadd__(self, other: "M") -> "M":
        return M(self.v + other.v)


def in_place_dunder() -> None:
    # CPython tries __iadd__ before __add__; both were defined and the binary
    # one ran.
    x = M(1)
    x += M(2)
    print(x.v)


def binary_dunder_only() -> None:
    y = M(1)
    y = y + M(2)
    print(y.v)


def set_in_place_through_an_alias() -> None:
    # `a |= {9}` rebound a fresh set, so every alias kept the old one.
    a: set[int] = {1}
    b: set[int] = a
    a |= {9}
    print(sorted(a), sorted(b))


def set_all_four() -> None:
    s: set[int] = {1, 2, 3}
    s -= {2}
    s &= {1, 3, 5}
    s ^= {3, 7}
    print(sorted(s))


def loop_target_survives() -> None:
    # CPython leaves the target bound after the loop.
    i = -1
    for i in range(3):
        pass
    print(i)


def loop_target_zero_trips() -> None:
    i = -1
    for i in range(0):
        pass
    print(i)


def loop_target_str() -> None:
    x = "z"
    for x in ["a", "b"]:
        pass
    print(x)


def loop_target_and_accumulator() -> None:
    i = -1
    total = 0
    for i in range(4):
        total += i
    print(i, total)


def main() -> None:
    in_place_dunder()
    binary_dunder_only()
    set_in_place_through_an_alias()
    set_all_four()
    loop_target_survives()
    loop_target_zero_trips()
    loop_target_str()
    loop_target_and_accumulator()


main()
