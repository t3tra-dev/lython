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
print(int("ff", base=16), int("11", base=2), int("  0x1f  ", 16))
print(int("ff", 16) == int("ff", base=16))
