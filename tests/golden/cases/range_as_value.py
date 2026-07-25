# A range used as a *value* rather than only as a for-loop source. len(r),
# r[i] and `v in r` are promised by range's `base_names = ["Sequence"]`, not by
# its own method_names, and had no implementation behind them. Output is
# CPython 3.14's.

r = range(5)
print(len(r))
print(r[0], r[2], r[-1])
print(3 in r, 7 in r, -1 in r, 5 in r)

# Explicit start/stop and a stride.
print(len(range(1, 4)), len(range(1, 6, 2)), len(range(1, 5, 2)))
s = range(2, 9, 3)
print(s[0], s[1], s[2])
print(7 in range(1, 10, 3), 6 in range(1, 10, 3))

# A descending range: the length, the indexing and the stride test all flip.
d = range(10, 0, -3)
print(len(d))
print(d[0], d[1], d[3])
print(7 in d, 8 in d)

# Empty in both directions.
print(len(range(5, 1)), len(range(0, 10, -1)))

# Iteration still agrees with indexing.
total: int = 0
for v in range(1, 10, 2):
    total = total + v
print(total)
