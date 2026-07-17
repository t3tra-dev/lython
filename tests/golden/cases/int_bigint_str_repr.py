# Cross-track (Wave 0 integration): big-int rendering must agree across
# print(x), str(x), and repr(x) (CPython: str(int) == repr(int)), including
# as pieces of adaptive-width string concatenation and multi-argument print.

big = 2 ** 90 + 12345
print(big)
print(str(big))
print(repr(big))
print(str(big) == repr(big))
neg = -(10 ** 25) - 7
print(neg, str(neg), repr(neg))
print(len(str(2 ** 100)))
print(str(0), str(-1), str(9223372036854775808))
print(repr("Ω" + str(big)))
print("value: " + str(big))
# Literals long enough that the 4-bits-per-digit parse width exceeds the
# 30-bit limb target (the zextOrTrunc regression the fuzz corpus caught).
long_lit = 10000000000000000000000000000000000000000000000000000000000000000000000123456789
print(long_lit)
print(str(-10000000000000000000000000000000000000000000000000000000000000000000000123456789))
