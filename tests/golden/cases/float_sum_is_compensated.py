# What this pins: `sum()` over floats carries CPython's compensation term.
#
#     print(sum([0.1, 0.2, 0.3]))      # 0.6000000000000001, CPython says 0.6
#     print(sum([1e100, 1.0, -1e100])) # 0.0, CPython says 1.0
#
# CPython's builtin_sum has used Neumaier summation for floats since 3.12: it
# keeps a correction term beside the running total and adds it back at the end.
# The naive fold loses the small term entirely, which the 1e100 line shows is
# not a rounding quibble -- the answer is off by the whole value.
#
# The correction is written as the same synthesized Python the rest of the fold
# is, so the arithmetic is this compiler's own float operations rather than a
# second implementation of them: t = acc + x, then c += (acc - t) + x when
# |acc| >= |x| and c += (x - t) + acc otherwise, then acc = t, and acc + c once
# the loop ends.
#
# Why this must run: every line here is a value that differs only in its last
# bits, or in whether a term survived at all. Nothing about the types changes.
#
# ⛔ `0.1 + 0.2 + 0.3` is NOT compensated and still prints 0.6000000000000001,
# as CPython's does: the compensation is sum()'s, not addition's. Both are here
# so the difference is visible.
#
# ⛔ AND AN EMPTY FLOAT ITERABLE STILL DIFFERS: `sum(empty)` prints 0.0 here and
# 0 in CPython, whose accumulator is the int start it never added to. Matching
# that needs one accumulator with two types. It is left out of this file rather
# than pinned wrong.
xs = [0.1, 0.2, 0.3]
print(sum(xs), 0.1 + 0.2 + 0.3)
print(sum([1e100, 1.0, -1e100]))
print(sum([0.1] * 10), sum([0.1] * 1000))
print(sum([1e16, 1.0, -1e16]), sum([1.0, 1e-16, -1.0]))
print(sum(xs, 0.0), sum(xs, 1.0), sum([2.5], 2.5))
print(sum([1, 2, 3]), sum([1, 2, 3], 10), sum(range(5)))

total = 0.0
i = 0
while i < 100:
    total = total + sum([0.1, 0.2, 0.3])
    i += 1
print(round(total, 9), abs(total - 60.0) < 1e-9)
