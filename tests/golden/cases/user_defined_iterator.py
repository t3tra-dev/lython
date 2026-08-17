# What this pins: a class that IS the iterator protocol -- `__iter__` returning
# self and `__next__` raising StopIteration.
#
#     for v in Count(4):
#     # runtime manifest has no Count.__next__ method
#
# The loop resolved `__iter__` through the class and then asked the MANIFEST
# for `__next__`, which a source class has no entry in -- so the protocol's own
# shape could not be iterated at all, in a for statement, a comprehension or a
# reducer.
#
# The protocol IS a try/except, so that is what the loop is written as:
#
#     __lyiter = <source>.__iter__()
#     while not __lydone:
#         try:
#             <target> = __lyiter.__next__()
#         except StopIteration:
#             __lydone = True
#         else:
#             <body>
#
# -- ordinary surface the walk already compiles, so `__next__` is inlined the
# way every other source method is and StopIteration goes through the real
# handler. Three details are forced rather than chosen: the handler sets a FLAG
# because `break` out of an except handler is unsupported below the emitter;
# the body goes in the ELSE because a name bound inside a try does not escape
# to the statements after it (`for v in it:` left `v` unresolved one line
# later); and the while tests the flag.
#
# Why this needs to run rather than assert on a diagnostic: the rewrite decides
# how many times `__next__` runs and what the target holds on the last pass.
# One iteration too many reads past the end, one too few drops an element, and
# both compile. `Pairs` below returns a tuple so the target unpacking is
# checked too.
#
# ⛔ `sum()`, `max()`, `min()`, `any()` and `all()` keep the old refusal. They
# desugar into their own synthesized loop -- a seen-flag switch over scratch
# names -- and running that through this rewrite produced "empty block: expect
# at least a terminator", a crash report where the old path gives a diagnostic.
# The emitter marks its reducer loops so they stay on it. `list()`, `sorted()`
# and the comprehensions are not reducers and do work.
#
# ⛔ A `break` or `continue` the body writes keeps the old refusal. It would
# leave the try's else, and that is unsupported below the emitter -- with a
# carried local it is "break/continue through try/finally", and without one it
# reached the lowering as a block with no terminator. So does a `for/else`:
# `while/else` cannot tell the exhaustion exit from the body's own break.
#
# Every expected line is python3.14's.


class Count:
    def __init__(self, n: int) -> None:
        self.n = n
        self.i = 0

    def __iter__(self) -> "Count":
        return self

    def __next__(self) -> int:
        if self.i >= self.n:
            raise StopIteration
        v = self.i
        self.i += 1
        return v


class Chars:
    def __init__(self, s: str) -> None:
        self.s = s
        self.i = 0

    def __iter__(self) -> "Chars":
        return self

    def __next__(self) -> str:
        if self.i >= len(self.s):
            raise StopIteration
        c = self.s[self.i]
        self.i += 1
        return c


class Pairs:
    def __init__(self, n: int) -> None:
        self.n = n
        self.i = 0

    def __iter__(self) -> "Pairs":
        return self

    def __next__(self) -> tuple[int, int]:
        if self.i >= self.n:
            raise StopIteration
        v = self.i
        self.i += 1
        return (v, v * v)


# --- the for statement, including the empty and one-element cases ----------
total = 0
for v in Count(4):
    total += v
print(total)

seen: list[int] = []
for v in Count(0):
    seen.append(v)
print(seen)

for v in Count(1):
    print("one:", v)


# --- comprehensions and reducers ------------------------------------------
print([x for x in Count(3)])
print(list(Count(4)))
print(sorted(Count(3), reverse=True))
print({v: v * 2 for v in Count(2)})
print(sorted({v % 2 for v in Count(4)}))


# --- a str element, so the element type is checked -------------------------
print([c.upper() for c in Chars("abc")])
print("-".join([c for c in Chars("xy")]))
for c in Chars("hi"):
    print(c)


# --- a tuple element, unpacked by the target -------------------------------
for a, b in Pairs(3):
    print(a, b)
print([a + b for a, b in Pairs(3)])


# --- the iterator is CONSUMED: a second pass sees nothing ------------------
once = Count(3)
print(list(once), list(once))
