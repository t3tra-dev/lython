# What this pins: `str(b, "utf-8")` -- CPython's bytes-and-encoding form of
# str() -- and that the one-argument spelling is still the repr.
#
#     print(str(b"ab", "utf-8"))
#     # static type !py.contract<"builtins.str"> does not provide manifest
#     # method '__init__'
#
# The message named str, and str is not what was unsupported: the second
# argument makes this a DECODE, which the runtime has had in all three arities
# all along. Falling through to the class-instantiation path is what produced a
# diagnostic about the target type instead of about the call.
#
# Why this must run: the answer is the decoded text. A multi-byte code point is
# here because a decode that is really a memcpy prints the same thing for pure
# ASCII -- "hé" is two bytes in and one character out, and len says which
# happened. The error path is the other half: an undecodable byte has to raise
# UnicodeDecodeError, not return a replacement.
#
# ⛔ `str(5, "utf-8")` is refused statically here where CPython raises TypeError
# at run time. That is this project's rule rather than a gap: the argument type
# is known at the call.
b = b"h\xc3\xa9"

print(str(b"ab", "utf-8"), str(b"ab", "utf-8", "strict"))
print(str(b, "utf-8"), len(str(b, "utf-8")), len(b))
print(str(b"ab"))

try:
    print(str(b"\xff", "utf-8"))
except UnicodeDecodeError:
    print("caught")
