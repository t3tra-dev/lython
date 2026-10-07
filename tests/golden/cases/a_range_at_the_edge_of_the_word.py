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

# The edge of the 64-bit word -- spelled out, not sys.maxsize, which is the
# 32-bit word on a 32-bit target.
WORD_MAX = 2**63 - 1
step = WORD_MAX // 2 + 1
print(range(5, WORD_MAX), len(range(WORD_MAX - 1, WORD_MAX)))
print(range(0, 10, WORD_MAX), list(range(0, 10, WORD_MAX)))
r = range(0, WORD_MAX, step)
print(list(r), len(r), list(iter(r)), r[-1], step in r, step + 1 in r)
q = range(0, -WORD_MAX - 1, -step)
print(list(q), sum(1 for _ in q), q[1], -step in q)
w = range(-WORD_MAX, WORD_MAX, WORD_MAX)
print(list(w), len(w), w[1], w[-2], WORD_MAX - 1 in w, 0 in w, -WORD_MAX in w)
huge = range(-WORD_MAX, WORD_MAX)
print(huge[0], huge[-1], huge[WORD_MAX], 0 in huge, 2**70 in huge, 2**70 in range(10))
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
