# What this pins: `int(s, base=16)`, the spelling CPython's own signature
# invites, beside the positional one it already accepted.
#
#     print(int("ff", base=16))
#     # static type !py.contract<"builtins.int"> does not provide manifest
#     # method '__init__'
#
# The two-argument int() is intercepted before the class-instantiation paths
# claim builtins.int, and the interception took the base POSITIONALLY -- it
# declined the moment a keyword appeared, so the call fell through to
# construction and the diagnostic talked about int's missing __init__ rather
# than about the argument.
#
# Why this must run: the answer is a parsed value, and both spellings must reach
# the SAME helper. A base of 2 and a base of 16 are here so the digit set is
# actually consulted, and the whitespace-and-prefix form is here because the
# helper strips and re-checks around the prefix -- a rewrite that forwarded the
# keyword to the wrong parameter would still print a number for "ff".
#
# ⛔ base=0 IS THE SAME PARSE WITH THE RADIX READ OFF THE PREFIX, which is why
# it is a variable and not a second function -- and it is not merely a default,
# because a bare leading zero is an ERROR under it: CPython reads `int("012",
# 0)` as an ambiguity with the old octal spelling and refuses it, while
# `int("00", 0)` and `int("0_0", 0)` are 0. It used to raise at run time saying
# so ("int() base 0 (auto-detect) is not supported"). The error messages keep
# saying "with base 0", because that is the base the caller passed.
print(int("ff", base=16), int("11", base=2), int("  0x1f  ", 16))
print(int("ff", 16) == int("ff", base=16))
print(int("0b101", 0), int("0o17", 0), int("0x1f", 0), int("12", 0))
print(int("0X1F", 0), int("-0x10", 0), int(" -0o17 ", 0), int("0b_1", 0))
print(int("0", 0), int("00", 0), int("0_0", 0), int("10", 0))
for bad in ["012", "0_1", "0x", "0b12"]:
    try:
        print(int(bad, 0))
    except ValueError as e:
        print("refused", e)
