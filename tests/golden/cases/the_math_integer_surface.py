# WHAT: the `math` functions that answer about INTEGERS and the three float
# predicates -- gcd, lcm, isqrt, factorial, comb, perm, isnan, isinf,
# isfinite, pow -- with their edge arguments and CPython's own error text.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: every one of them is a
# VALUE, and the edges are where a kernel written from the formula and a kernel
# written from CPython's loop diverge: gcd of a negative, lcm through a zero,
# comb past n, isqrt at a perfect square and one below it.
#
# ⛔ THESE READ THE i64 WINDOW. `builtins.int` is arbitrary precision and these
# kernels are not: past the machine window they raise OverflowError through the
# same `unbox.i64` every other machine-word kernel here uses. `math.factorial`
# is the one a reader will notice -- CPython answers 21! and this does not.
#
# ⛔ `math.fsum` IS DELIBERATELY ABSENT: CPython's is exact, and a naive loop
# agrees on two terms and diverges on longer ones.
import math

print(math.gcd(12, 18), math.gcd(0, 5), math.gcd(-4, 6), math.gcd(7, 0), math.gcd(0, 0))
print(math.lcm(4, 6), math.lcm(0, 5), math.lcm(-4, 6), math.lcm(3, 3))
print(math.isqrt(0), math.isqrt(1), math.isqrt(15), math.isqrt(16), math.isqrt(17))
print(math.isqrt(10000), math.isqrt(99999999))
print(math.factorial(0), math.factorial(1), math.factorial(5), math.factorial(12))
print(math.comb(5, 2), math.comb(5, 0), math.comb(5, 5), math.comb(2, 5), math.comb(20, 10))
print(math.perm(5, 2), math.perm(5, 0), math.perm(5, 5), math.perm(2, 5))
print(math.isnan(math.nan), math.isnan(1.0), math.isinf(math.inf), math.isinf(1.0))
print(math.isfinite(1.0), math.isfinite(math.inf), math.isfinite(math.nan))
print(math.pow(2.0, 10.0), math.pow(9.0, 0.5), math.pow(2.0, -1.0))

for bad in ["factorial", "isqrt", "comb-n", "comb-k", "perm-n", "perm-k"]:
    try:
        if bad == "factorial":
            math.factorial(-1)
        elif bad == "isqrt":
            math.isqrt(-1)
        elif bad == "comb-n":
            math.comb(-1, 2)
        elif bad == "comb-k":
            math.comb(5, -1)
        elif bad == "perm-n":
            math.perm(-1, 2)
        else:
            math.perm(5, -1)
        print(bad, "no error")
    except ValueError as e:
        print(bad, "ValueError:", e)
