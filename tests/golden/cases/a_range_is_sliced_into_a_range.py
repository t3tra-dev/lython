# WHAT: slicing a range gives a range, as CPython's compute_slice does -- the
# slice's indices clamped against the range's length, each mapped through the
# range, the steps multiplied, and the clamped stop kept in the repr
# (`range(10)[1:8:3]` is `range(1, 8, 3)`, not the `range(1, 10, 3)` a length
# would give) -- and a zero step is a ValueError.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answers are reprs,
# lengths and elements of ranges computed at run time from the slice; a slice
# that keeps the wrong stop compiles exactly like a right one.
import sys

r = range(10)
print(r[2:5], r[::2], r[::-1], r[1:8:3], r[-3:], r[:-3])
print(r[5:2], r[100:], r[-100:2], r[::-3], r[8:1:-2], r[-1:-11:-1])
print(range(0, 20, 3)[::2], range(20, 0, -3)[1::2], range(5, 50, 5)[::-2])
print(range(-5, 5)[3:-3], range(10, -10, -4)[1:], range(1, 2)[5:])
print(list(range(10)[7:2:-2]), len(range(1, 100, 7)[3:11:2]), 6 in range(0, 20, 3)[::2])
print(range(-sys.maxsize, sys.maxsize, sys.maxsize)[1:], range(sys.maxsize)[-2:])
try:
    print(r[::0])
except ValueError as e:
    print("ValueError", e)
