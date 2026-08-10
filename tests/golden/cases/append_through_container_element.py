# Why execution: the second append destroyed the first. `g[0].append(9)` then
# `g[0].append(8)` stored both at the same index and moved the length past a
# slot nobody wrote -- the JIT aborted inside repr, and the AOT build printed
# `[1, 8, None]` with exit 0. Only running tells those from the right list.
#
# The cause is that `g[0]` rebuilds the inner list's description from the outer
# list's element map on every read, so the second append saw the same
# one-element snapshot as the first. A receiver the walk re-derives per read is
# not one whose mutations it has all seen, so it goes to the runtime arm, which
# loads the length and stores there.


def repeated_appends() -> None:
    grid: list[list[int]] = [[1]]
    grid[0].append(9)
    grid[0].append(8)
    print(grid[0])


def three_of_them() -> None:
    grid: list[list[int]] = [[1]]
    grid[0].append(2)
    grid[0].append(3)
    grid[0].append(4)
    print(grid[0])


def separate_elements() -> None:
    grid: list[list[int]] = [[1], [2]]
    grid[0].append(9)
    grid[1].append(8)
    print(grid)


def dict_value() -> None:
    groups: dict[str, list[int]] = {"k": [1]}
    groups["k"].append(2)
    groups["k"].append(3)
    print(groups["k"])


def strings() -> None:
    rows: list[list[str]] = [["a"]]
    rows[0].append("b")
    rows[0].append("c")
    print(rows[0])


def main() -> None:
    repeated_appends()
    three_of_them()
    separate_elements()
    dict_value()
    strings()


main()
