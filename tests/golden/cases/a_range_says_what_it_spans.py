# What: `.start`, `.stop` and `.step` read words the range object stores, so
# the values have to come back as the constructor's arguments -- including the
# ones it defaults. Running it is the only way to see which word each name
# reads; all three are ints, so a swapped slot is a wrong number.
def span(r: range) -> int:
    return r.stop - r.start


print(range(3).start, range(3).stop, range(3).step)
print(range(2, 9).start, range(2, 9).stop, range(2, 9).step)
print(range(1, 9, 2).start, range(1, 9, 2).stop, range(1, 9, 2).step)
print(range(5, 1, -1).start, range(5, 1, -1).step)
print(span(range(3)), span(range(2, 9, 3)))

r = range(0, 10, 2)
print(len(r), r[2], list(r), 4 in r)
