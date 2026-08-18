# What this pins: int() over bytes -- the value, the sign, the surrounding
# whitespace, the digit separator, and the ValueError's b-prefixed repr.
#
#     print(int(b"12"))
#     # static type !py.contract<"builtins.int"> does not provide manifest
#     # method '__init__'
#
# int(x) is intercepted before the class-instantiation paths claim builtins.int,
# and the interception knew int / bool / str / float; bytes fell through to
# instantiation, so the diagnostic reported a missing `int.__init__` -- about the
# TARGET type -- when what was unsupported was the argument. CPython takes bytes
# anywhere int() takes str.
#
# Why this must run: the answer is a parsed value. The scan is the same one str
# goes through (whitespace-trimmed span, optional sign, interior single
# underscores, arbitrary length via the limb accumulate), so what needs
# executing is that the bytes payload reaches it and the digits come back --
# including past 2**64, where a wrong limb walk is still a plausible number.
#
# The failure path is the reason the parse is a shared helper and the raise is
# not: CPython reports the repr of the offending object, and bytes reprs itself
# with a b prefix. Splitting the scan out and giving each caller its own raiser
# is what keeps `int(b"ab")` from claiming a str failed.
b = b"0012"
print(int(b))
print(int(b[0:2]), int(b"-3"), int(b" 8 "), int(b"1_0"))
print(int(b"999999999999999999999999") + 1)
try:
    print(int(b"ab"))
except ValueError as e:
    print("caught", e)
