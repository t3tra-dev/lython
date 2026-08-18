# What this pins: the eight math functions added beside the nine that were
# there, and -- for each one that has one -- the error CPython raises instead of
# a silent inf/nan.
#
#     import math
#     print(math.log2(8.0))
#     # module 'math' has no attribute 'log2' in this runtime
#
# Why this must run: the point of each of these is the BITS. log2 / log10 /
# exp2 / atan2 / fmod / copysign are libm's, which is where CPython gets them
# too, so the printed repr is the check that the kernel calls the operation it
# names and unboxes the operands in the right ORDER -- atan2 is (y, x) and fmod
# is (x, y), and a swap still prints a plausible number. degrees and radians are
# the two CPython computes as a single multiply by a constant rounded once, so
# the repr also checks that this does not divide by pi instead.
#
# The domain and range checks are the other half. CPython's math_1 wrapper reads
# errno and raises: log2(0.0) and log10(-1.0) name the constraint and the
# operand, fmod(1.0, 0.0) is the generic "math domain error", exp2(10000.0) is
# an OverflowError. Returning -inf, nan or inf silently is what this refuses to
# do, and only running it says which happened.
#
# ⛔ fmod's rule is CPython's own -- a NaN result from operands that were not
# NaN -- rather than a list of cases, which is why fmod(nan, 1.0) still returns
# nan here and fmod(inf, 2.0) raises.
#
# isclose comes with them and is not libm's: it is CPython's math_isclose
# verbatim, and every line of that is load-bearing -- `a == b` first so an
# infinity is close to itself, the isinf check so inf against a finite number is
# False rather than inf <= inf, BOTH relative tests because the tolerance is
# relative to either operand, and no NaN case at all, since every comparison is
# false for a NaN and that is the answer. Its tolerances are keyword-only, which
# is why the manifest declares them under `kwonly`: `isclose(a, b, 0.2)` is a
# TypeError in CPython and is refused here too.
#
# ⛔ hypot and dist are still missing, and now measured rather than asserted:
# libm's hypot disagrees with CPython's on 32,071 of 200,000 random pairs (1
# ulp, worst relative 2.2e-16), because CPython 3.14 accumulates the squares in
# double-double instead of calling libm. Wiring libm would print a different
# last digit for one input in six.
#
# The keyword spellings run because the call carries the mapping: the contract's
# arg_names followed by its kw_names are the parameter order, which is the order
# the manifest function declares its inputs in, so a keyword resolves to a
# position and the positions nobody supplied take the function's own default.
# Both orders are here because a keyword call is order-free and the operand walk
# is not.
#
# ⛔ Also missing: gcd / lcm / comb / perm / prod are int work rather than libm
# calls.
import math

print(math.log2(8.0), math.log10(100.0), math.exp2(3.0), math.exp2(0.5))
print(math.atan2(1.0, 1.0), math.atan2(-0.0, -1.0))
print(math.fmod(7.0, 3.0), math.fmod(-7.0, 3.0), math.fmod(7.0, -3.0))
print(math.fmod(math.nan, 1.0), math.fmod(2.0, math.inf))
print(math.copysign(3.0, -0.0), math.copysign(-2.5, 1.0))
print(math.degrees(1.0), math.radians(1.0), math.degrees(math.pi))

print(math.isclose(1.0, 1.0), math.isclose(1.0, 1.0000000001))
print(math.isclose(1.0, 1.1), math.isclose(0.1 + 0.2, 0.3))
print(math.isclose(math.inf, math.inf), math.isclose(math.nan, math.nan))
print(math.isclose(1.0, 1.1, rel_tol=0.2), math.isclose(1.0, 1.1, abs_tol=0.2))
print(math.isclose(1.0, 1.1, rel_tol=0.0, abs_tol=0.2),
      math.isclose(1.0, 1.1, abs_tol=0.2, rel_tol=0.0))
print(math.isclose(0.0, 1e-12, abs_tol=1e-9), math.isclose(0.0, 1e-8, abs_tol=1e-9))

try:
    math.isclose(1.0, 1.1, rel_tol=-1.0)
except ValueError as e:
    print("isclose:", e)

try:
    math.log2(0.0)
except ValueError as e:
    print("log2:", e)
try:
    math.log10(-1.0)
except ValueError as e:
    print("log10:", e)
try:
    math.fmod(1.0, 0.0)
except ValueError as e:
    print("fmod:", e)
try:
    math.exp2(10000.0)
except OverflowError as e:
    print("exp2:", e)
