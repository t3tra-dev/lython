# A beyond-i64 `int` literal built 300,000 times inside one frame, plus the limb
# arithmetic that says the limbs are right.
#
# What this pins, and why it is a separate case from
# str_literal_loop_frame_growth: `lowerIntConstant` splits a beyond-i64 literal
# into 30-bit limbs at compile time and used to stage them in a per-execution
# `memref.alloca<?xi32>`. An alloca outside the entry block is not
# `AllocaInst::isStaticAlloca()`, so it is a runtime stack adjustment reclaimed
# only at function return, and the frame grew 4 bytes per limb per iteration until
# the stack guard raised RecursionError. `LyLong_FromDigits` allocates its own
# digit array through `__ly_long_alloc_raw` and copies out of what it is given,
# so the block is the initializer's argument and belongs in read-only data.
#
# This case is red on the build before the `str`/`bytes` fix AND on the build
# after it: the two literals grew the same frame and the shorter one hit the cliff
# first, so the int instance only became visible once the str one was gone. That
# is the reason to keep both cases rather than merging them.
#
# The value checks are not decoration. Two occurrences of one literal now share
# one read-only global, so a wrong limb, a lost sign, or a block reused for the
# wrong literal all present as a wrong number rather than as a crash. Every line
# below is byte-identical to CPython 3.14.
a = 123456789012345678901234567890123456789012345678901234567890
b = -98765432109876543210987654321098765432109876543210987654321
print(a)
print(b)
print(a + b)
print(a * b)
print(a % 1000000007, b % 1000000007)
print(a // 12345678901234567890, a > b, -a < b)

# Sharing must be observable only as identical values: `d` differs from `a` in
# its last digit and has the same limb count, so a name collision between the two
# globals would print `0` for `d - c`.
c = 123456789012345678901234567890123456789012345678901234567890
d = 123456789012345678901234567890123456789012345678901234567891
print(a == c, a == d, d - c, c - a)

# The 30-bit limb width and the i64 cliff, from both sides.
print(1073741824, 1073741823, 1152921504606846976)
print(9223372036854775807, 9223372036854775808, 18446744073709551616)
print(-9223372036854775808, -9223372036854775809)
print(170141183460469231731687303715884105727)


def in_a_loop(n: int) -> int:
    total = 0
    i = 0
    while i < n:
        big = 55555555555555555555555555555555555555555555555
        total = total + (big % 1000003)
        i = i + 1
    return total


print(in_a_loop(300000))
