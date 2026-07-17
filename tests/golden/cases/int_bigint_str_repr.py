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
