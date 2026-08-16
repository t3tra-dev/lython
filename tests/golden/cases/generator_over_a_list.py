# What this pins: a generator whose body iterates a LIST (or a tuple).
#
#     def each(xs: list[int]):
#         for x in xs:
#             yield x
#     # source generator next lowering currently supports yields whose runtime
#     # value is a single lane, and 'builtins.int' has 3
#
# A `for` loop keeps its position in a function-local cell, and a cell cannot
# survive a suspension -- so the state machine declines the body and it falls
# to the non-suspending path, which carries one lane per yield. An int read out
# of a list is the 3-lane object form, so it lands there. `for i in range(n):
# yield i` has always run, because a range element rides an i64 frame lane.
#
# The rewrite is the one the lazy-iterator VALUE synthesis already uses for
# exactly this reason ("Bodies use index-based while loops on purpose"),
# applied to the loop the program wrote: an int index rides a frame lane where
# an iterator's position cannot. CPython's list iterator is an index too, so a
# mutation during iteration observes the same elements.
#
# Why this needs to run rather than assert on a diagnostic: the rewrite moves
# the increment BEFORE the body so `continue` still advances, and maps the for's
# `else` onto the while's. A loop that advanced after the body would hang on a
# `continue`, and one that lost the else would print a shorter list -- both
# compile. The break/continue/else cases below are what say the rewrite kept
# the loop's shape.
#
# ⛔ Only list and tuple sources. A str or a range already has a manifest
# iterator that survives (`str_iterator`, `range_iterator`), a dict or a set
# has no index, and rewriting a loop that already works would trade a working
# path for an untested one.
#
# Every expected line is python3.14's.


def each(xs: list[int]):
    for x in xs:
        yield x


def doubled(xs: list[str]):
    for x in xs:
        yield x * 2


def over_tuple(t: tuple[int, int, int]):
    for x in t:
        yield x * 2


print(list(each([1, 2, 3])), sum(each([4, 5])))
print(list(doubled(["a", "bb"])))
print(list(over_tuple((1, 2, 3))))
print(list(each([])))


# --- continue, break and else, which the rewrite has to keep --------------
def skipping(xs: list[int]):
    for x in xs:
        if x == 2:
            continue
        yield x


def upto(xs: list[int], k: int):
    for x in xs:
        if x > k:
            break
        yield x
    else:
        yield -1


print(list(skipping([1, 2, 3])))
print(list(upto([1, 2, 5], 3)), list(upto([1, 2], 9)))


# --- a loop-carried local, and a nested loop -------------------------------
def counted(xs: list[int]):
    n = 0
    for x in xs:
        n += 1
        yield x + n


def flattened(rows: list[list[int]]):
    for r in rows:
        for c in r:
            yield c


print(list(counted([10, 20])))
print(list(flattened([[1, 2], [], [3]])))


# --- a field, which is where this was found --------------------------------
class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs


def from_field(b: Bag):
    for x in b.xs:
        yield x


total = 0
for v in from_field(Bag([1, 2, 3])):
    total += v
print(total)


# --- THE CONTROL: range and str, which always worked -----------------------
def counting(n: int):
    for i in range(n):
        yield i


def letters(s: str):
    for c in s:
        yield c


print(list(counting(4)))
print(list(letters("abc")))
