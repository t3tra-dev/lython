# What this pins: a class with `__len__` and `__getitem__` and no `__iter__`.
#
#     for v in Seq([1, 2, 3]):
#     # static type !py.contract<"Seq"> does not provide manifest method
#     # '__iter__'
#
# That is Python's fallback iteration protocol: `iter()` on an object with no
# `__iter__` but with `__getitem__` walks indices from 0 until IndexError, and
# every sequence written before `__iter__` existed relies on it. `s[0]` and
# `len(s)` both worked here; only iterating did not.
#
# The fallback IS the index loop this compiler already builds for a generator's
# list iteration, so it is the same rewrite reached for a different reason.
# Three walks had to learn the rule, because three of them ask the iteration
# question: the `for` statement (which runs it), `iterationElementType` (a
# reducer's accumulator), and the comprehension walk (which needs the iterator
# type as well, so it asks the two questions itself).
#
# Why this needs to run rather than assert on a diagnostic: the element type
# comes from `__getitem__`'s RESULT, and picking it wrong compiles -- a str
# sequence whose element typed as int would fail only where the value is used.
# The two classes below have different element types for that reason, and the
# reducers over them check the values rather than the shape.
#
# ⛔ A class that HAS `__iter__` is iterated through it, which is where a
# stateful iterator's own position lives. CPython's fallback has the same
# precedence, and the last class below is the control: its `__getitem__` would
# give 100, 200 and its `__iter__` gives 7, 8.
#
# Every expected line is python3.14's.


class Seq:
    def __init__(self, xs: list[int]) -> None:
        self.xs = xs

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, i: int) -> int:
        return self.xs[i]


class Words:
    def __init__(self, ws: list[str]) -> None:
        self.ws = ws

    def __len__(self) -> int:
        return len(self.ws)

    def __getitem__(self, i: int) -> str:
        return self.ws[i]


s = Seq([1, 2, 3])
print(len(s), s[0], s[2])

# --- the for statement -----------------------------------------------------
total = 0
for v in s:
    total += v
print(total)

# --- comprehensions, which ask the question their own way ------------------
print([v for v in s])
print({v: v * 2 for v in s})
print(sorted({v % 2 for v in s}))

# --- the reducers, which need the element type -----------------------------
print(sum(s), max(s), min(s))
print(sorted(s, reverse=True))
print(list(s))

# --- a str element, so the element type is checked and not assumed ---------
w = Words(["a", "bb"])
print([x.upper() for x in w])
print("-".join([x for x in w]))
for x in w:
    print(x)


# --- an empty one ----------------------------------------------------------
empty = Seq([])
print(len(empty), [v for v in empty], sum(empty))


# --- THE CONTROL: __iter__ wins when the class has one ---------------------
class WithIter:
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i * 100

    def __iter__(self):
        return iter([7, 8])


both = WithIter(3)
print([v for v in both])
print(both[1], len(both))
