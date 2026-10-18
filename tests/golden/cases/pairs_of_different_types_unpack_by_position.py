# WHAT: `for a, b in [(1, 1), ("s", "t"), ([1], [1]), ...]` -- test data of
# pairs whose types differ from pair to pair, as a parametrized assertEqual
# loop writes it -- unpacks each pair; so does a pair a function returns as
# `tuple[int, int] | tuple[str, str]`.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: each pair's types are the
# run-time ones the comparison and the repr read. The list was a list of a
# union of tuples, which has no `__getitem__` ("does not provide manifest
# method '__getitem__'"), and the function's union could not be unpacked
# either.
class P:
    def __init__(self, x: int) -> None:
        self.x = x

    def __eq__(self, other: object) -> bool:
        return isinstance(other, P) and other.x == self.x

    def __repr__(self) -> str:
        return "P(%d)" % self.x


def assert_equal(first: object, second: object) -> str:
    if not first == second:
        return "%r != %r" % (first, second)
    return "ok"


for a, b in [(1, 1), ("s", "s"), ([1, 2], [1, 2]), ((1, "a"), (1, "a")),
             ({"k": 1}, {"k": 1}), (P(1), P(1)), (1.5, 1.5), ([1], [2])]:
    print(assert_equal(a, b))


def pick(n: int) -> tuple[int, int] | tuple[str, str]:
    if n > 0:
        return (n, n + 1)
    return ("s", "t")


c, d = pick(1)
e, f = pick(0)
print(c, d, e, f)
