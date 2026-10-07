# WHAT: a range whose bounds reach the ends of the 64-bit word answers as
# CPython: `range(5, sys.maxsize)` is that range (the constructor read 2**63 - 1
# as "argument absent" and built `range(5)`), a range longer than INT64_MAX
# indexes, tests membership and iterates (its length was measured in signed
# arithmetic and came out 0), iteration stops when the step after the last
# element would leave the word (it wrapped and never stopped), an int past the
# word is in no range, and len() of a range longer than INT64_MAX is CPython's
# OverflowError.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: every answer is computed
# from run-time words at the edge of their range; a wrapped length or a
# wrapped iterator compiles exactly like a right one.
import sys

step = sys.maxsize // 2 + 1
print(range(5, sys.maxsize), len(range(sys.maxsize - 1, sys.maxsize)))
print(range(0, 10, sys.maxsize), list(range(0, 10, sys.maxsize)))
r = range(0, sys.maxsize, step)
print(list(r), len(r), list(iter(r)), r[-1], step in r, step + 1 in r)
q = range(0, -sys.maxsize - 1, -step)
print(list(q), sum(1 for _ in q), q[1], -step in q)
w = range(-sys.maxsize, sys.maxsize, sys.maxsize)
print(list(w), len(w), w[1], w[-2], sys.maxsize - 1 in w, 0 in w, -sys.maxsize in w)
huge = range(-sys.maxsize, sys.maxsize)
print(huge[0], huge[-1], huge[sys.maxsize], 0 in huge, 2**70 in huge, 2**70 in range(10))
seen = 0
for v in huge:
    seen += 1
    if seen == 2:
        print("second", v)
        break
try:
    print(len(huge))
except OverflowError as e:
    print("OverflowError", e)
try:
    print(range(3)[2**70])
except IndexError as e:
    print("IndexError", e)
try:
    print(range(0, 10, 0))
except ValueError as e:
    print("ValueError", e)
