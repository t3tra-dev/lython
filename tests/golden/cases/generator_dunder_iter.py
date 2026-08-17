# What this pins: `__iter__` written as a GENERATOR.
#
#     class Pair:
#         def __iter__(self):
#             yield self.a
#             yield self.b
#     for v in Pair(1, 2):
#     # static type !py.literal<None> does not provide manifest method
#     # '__next__'
#
# It is the ordinary way to make a small object iterable, and the loop asked the
# class for `__iter__` and got its BODY result -- which for a generator is None.
# The value a CALL to it produces is the generator object, and iterating a
# generator already runs, so the loop iterates the call: the iterable becomes
# `<source>.__iter__()`, and the direct-call route takes the method symbol with
# the receiver as its leading positional.
#
# Why this needs to run rather than assert on a diagnostic: the generator's
# frame holds the receiver across every suspension, and each `yield` reads a
# field off it. A frame that dropped it would still compile and print the wrong
# member -- which is what `Pair` below checks by yielding two different fields.
#
# ⛔ A generator `__iter__` that iterates ANOTHER instance of its own class
# (`for x in k: yield x` over a list of children -- the tree traversal) is
# "source generator next lowering currently supports only straight-line": the
# nested loop over a source-class iterable becomes a try/except, which the state
# machine declines. A list field is fine, which is the second class below.
#
# Every expected line is python3.14's.


class Pair:
    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b

    def __iter__(self):
        yield self.a
        yield self.b


class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs = xs

    def __iter__(self):
        for x in self.xs:
            yield x


class Words:
    def __init__(self, s: str) -> None:
        self.s = s

    def __iter__(self):
        for w in self.s.split():
            yield w.upper()


# --- the straight-line generator, in every consumer -----------------------
p = Pair(1, 2)
print(list(p))
for v in p:
    print(v)
print(sum(p), max(p), min(p))
print(sorted(p, reverse=True))
print([v * 2 for v in p])
print({v: v for v in p})


# --- one that walks a list field ------------------------------------------
b = Bag([3, 1, 2])
print(list(b), sum(b))
for v in b:
    print(v)
print(sorted(b))
print(len([v for v in b]))


# --- and one whose elements are strings -----------------------------------
w = Words("a bb ccc")
print(list(w))
print([len(x) for x in w])


# --- the receiver's fields are read on each resume ------------------------
# `Pair` yields a THEN b, so a frame that lost the receiver between the two
# suspensions would repeat one of them.
print(list(Pair(10, 20)), list(Pair(-1, -2)))
print([x for x in Pair(7, 8)])
