# What this pins: `raise E(name)` where `name` is not a temporary.
#
#     i = 0
#     while i < 5:
#         try:
#             raise ValueError(i)
#         except ValueError:
#             pass
#         i += 1
#     # owned resource from @LyLong_FromI64 result 0 is released or
#     # transferred more than once on one CFG path
#
# The payload block retains its own reference to each argument AND released the
# argument's -- a pair that is right for a temporary, which has one token nobody
# else will discharge, and a double free for a value that outlives the raise:
# `i += 1` releases the old int and the loop releases it again. It aborted under
# --release, so the refusal was the affine verifier standing in front of a real
# defect rather than a strictness of its own.
#
# Why this must run: the fix is about WHOSE reference the block took, which only
# a program that keeps using the value afterwards can show. `total` sums the
# argument read back out of e.args, so a value freed under the block would be
# read after free rather than merely leaked, and the loop runs often enough for
# the allocator to hand the slot to someone else.
#
# ⛔ The temporary case still hands its token over, which is the other half:
# `raise ValueError(str(i))` and `raise ValueError("m", i)` build a fresh str
# per iteration and nothing else would ever release it. Both are here, and
# tests/leak_gate.py reads 0 for this file.
def sum_over(limit: int) -> int:
    total = 0
    i = 0
    while i < limit:
        try:
            raise ValueError(i)
        except ValueError as e:
            total += len(e.args)
        i += 1
    return total


i = 0
total = 0
while i < 40:
    try:
        raise ValueError(i)
    except ValueError as e:
        total += len(e.args)
    i += 1
print("loop carried", i, total)

j = 0
temporaries = 0
while j < 40:
    try:
        raise ValueError(str(j))
    except ValueError as e:
        temporaries += len(e.args)
    j += 1
print("temporary", temporaries)

k = 0
pairs = 0
while k < 40:
    try:
        raise ValueError("m", k)
    except ValueError as e:
        pairs += len(e.args)
    k += 1
print("pair", pairs)

n = 0
while n < 40:
    xs = [n, n + 1]
    try:
        raise ValueError(xs)
    except ValueError as e:
        pairs += len(e.args)
    n += 1
print("list argument", pairs)

m = 0
while m < 40:
    try:
        raise ValueError(m)
    except ValueError:
        break
print("break in the handler", m)

print("in a function", sum_over(40))
