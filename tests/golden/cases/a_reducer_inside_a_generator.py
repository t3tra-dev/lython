# What: `max` / `min` / `sum` / `sorted` inside a generator body.
#
# Two defects, both refusals of programs the same code outside a generator ran:
#   - the fold's running best rides a cond_br whose two successors are one
#     block; the generator's resume threads its carried locals through that
#     merge, and the lowering spliced the physical operands onto the wrong one
#     of the two edges ("non-lowered users ... py.add");
#   - `yield max(row)` over a loop target was typed with no context, so the
#     yield was None and the annotation refused ("yields None").
#
# Why execution: the values are what a mis-spliced edge would change.
from typing import Iterator


def g(xs: list[int], fs: list[float], ss: list[str]) -> Iterator[int]:
    print(max([4, 1]), min(xs))
    best = max(xs)
    print(best, max(fs), min(ss), max(xs, default=0))
    for row in [xs, [7, 8]]:
        print(max(row) - min(row))
        yield max(row)
    yield min(xs)


print(list(g([3, 9, 2], [1.5, -2.0], ["b", "a"])))


def per_row(rows: list[list[int]]) -> Iterator[int]:
    for row in rows:
        yield max(row)
        yield sum(row)


def sorted_rows(rows: list[list[str]]) -> Iterator[list[str]]:
    for row in rows:
        yield sorted(row)


print(list(per_row([[3, 9], [2]])), list(sorted_rows([["b", "a"], ["c"]])))
