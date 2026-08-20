# What this pins: `max(xs, default=...)` and `min(xs, default=...)`.
#
#     print(max(xs, default=0))
#     # max() with the 'default' keyword argument is not supported: it is
#     # folded over an iterable or over two or more operands, and neither form
#     # takes it
#
# The fold already emits a seen-flag and `if not seen: raise ValueError(...)`,
# so the default is that branch's other answer -- and the cheapest way to give
# it is not a second arm but a different SEED: the accumulator starts at the
# default and the empty guard disappears. Seeding the fabricated placeholder
# and assigning the default afterwards compiled and leaked, because the
# placeholder was only ever unread thanks to the empty path RAISING; give that
# path a return and the fabrication reaches the function exit.
#
# Why this must run: the empty case is the whole feature, and it is a value the
# program prints rather than a type. The loop at the end runs the fold often
# enough that a leaked seed shows -- tests/leak_gate.py reads 0 for this file.
#
# ⛔ `key=` and `default=` compose, in either order, because the two keywords
# are now read by name instead of by position.
xs = [5, 3, 1]
empty: list[int] = []
print(max(xs, default=0), min(xs, default=0))
print(max(empty, default=-1), min(empty, default=99))
print(max([], default=7), min([], default=8))

words = ["bb", "a", "ccc"]
no_words: list[str] = []
print(max(words, key=len, default=""), min(words, key=len, default=""))
print(max(no_words, key=len, default="none"), max(no_words, default="none"))
print(min(no_words, default="none", key=len))

rows = [(1, "a"), (3, "c")]
no_rows: list[tuple[int, str]] = []
print(max(rows, key=lambda p: p[0], default=(0, "z")))
print(max(no_rows, key=lambda p: p[0], default=(0, "z")))

try:
    print(max(empty))
except ValueError as e:
    print("still raises", e)

total = 0
i = 0
while i < 200:
    pair = [i, i + 1]
    total += max(pair, default=-1) + min(pair, default=-1)
    i += 1
print("loop", total)
