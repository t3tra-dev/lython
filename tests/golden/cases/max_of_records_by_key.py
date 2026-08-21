# `max(rows, key=lambda r: r.score)` over a list of INSTANCES -- what every
# "pick the best record" line looks like -- was refused: "max() needs an
# element type the fold can seed (int, str, float, bool, or a tuple of those)".
# `sorted(rows, key=...)` beside it has always worked. Must run: the values are
# what say WHICH element came back, and the tie rule (CPython returns the FIRST
# maximal element) is only visible in the answer.


class Row:
    def __init__(self, name: str, score: int) -> None:
        self.name = name
        self.score = score


rows = [Row("ann", 90), Row("bob", 75), Row("cara", 90), Row("dan", 75)]
print(max(rows, key=lambda r: r.score).name)
print(min(rows, key=lambda r: r.score).name)

# One element, and the loop over range(1, 1) never runs.
one = [Row("solo", 5)]
print(max(one, key=lambda r: r.score).name, min(one, key=lambda r: r.score).name)

# An empty one raises ValueError, not IndexError: the guard runs before the
# first element is read.
empty: list[Row] = []
try:
    print(max(empty, key=lambda r: r.score).name)
except ValueError as err:
    print("max:", err)
try:
    print(min(empty, key=lambda r: r.score).name)
except ValueError as err:
    print("min:", err)

# The key may be a named function and may return a str.
def by_name(r: Row) -> str:
    return r.name


print(max(rows, key=by_name).name, min(rows, key=by_name).name)

# The primitive folds are unchanged, with and without a key.
print(max([3, 1, 2]), min([3, 1, 2]), max([1, 2], default=9), max([], default=7))
print(max(["bb", "a", "ccc"], key=len), min(["bb", "a", "ccc"], key=len))
print(max([("a", 90), ("b", 75)], key=lambda p: p[1])[0])
