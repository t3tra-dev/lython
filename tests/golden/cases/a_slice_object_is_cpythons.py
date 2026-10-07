# WHAT: the slice object as sliceobject.c makes it -- one to three bounds kept
# as given, None for an absent one, read back as attributes; the repr; ==,
# != and the orderings as the (start, stop, step) tuples compare; a hash that
# makes it a dict key and a set member; and indices(length), clamped the way
# a sequence of that length reads the slice, with the step returned as the
# slice holds it, a length past the 64-bit word computed in Python ints, and
# its two ValueErrors.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: every answer is a value
# the object computes at run time from bounds made at run time. The ints past
# the word come from a call so that they are fresh objects, not immortal
# literals: a bound released before the slice took its own reference read
# freed memory, and only a value the program built can show it.


def big(n: int) -> int:
    return 2**n


# The edge of the 64-bit word -- spelled out, not sys.maxsize, which is the
# 32-bit word on a 32-bit target.
WORD_MAX = 2**63 - 1
s = slice(1, 10, 3)
print(s, s.start, s.stop, s.step)
print(repr(slice(7)), str(slice(None, None, -1)), f"{slice(2, None)}")
t = slice(big(70), -big(70))
print(t, t.start, t.stop, t.step)
if t.start is not None:
    print(t.start + 1)
print(slice(1, 2) == slice(1, 2, None), slice(1, 2) != slice(1, 3), slice(None) == slice(None))
print(slice(1, 2) < slice(1, 3), slice(2) > slice(1), slice(1, 2) <= slice(1, 2), slice(3, 4) >= slice(3, 5))
print(sorted([slice(3, 4), slice(1, 9), slice(1, 2)]))
print(hash(slice(1, 2)) == hash(slice(1, 2)), hash(slice(None, 3)) == hash(slice(3)))
keys = {slice(1, 2): "a", slice(None, 3): "b"}
print(keys[slice(1, 2)], keys[slice(3)], len({slice(1, 2), slice(1, 2, None), slice(2)}))
print(slice(1, 100, 3).indices(10), slice(None, None, -1).indices(10), slice(-3, None).indices(4))
print(slice(big(70)).indices(5), slice(-big(70), None, 2).indices(5), slice(1, 9, 2).indices(0))
print(slice(None, None, big(70)).indices(5), slice(None, None, -WORD_MAX - 1).indices(5))
print(slice(None).indices(big(70)), slice(-3, None).indices(big(70)), slice(None, None, -1).indices(big(70)))
print(slice(5, -5, -2).indices(big(70)), slice(big(75), None, -1).indices(big(70)))
for length in [-1, -big(70)]:
    try:
        print(slice(1).indices(length))
    except ValueError as e:
        print("ValueError", e)
try:
    print(slice(1, 2, 0).indices(5))
except ValueError as e:
    print("ValueError", e)
