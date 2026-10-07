# WHAT: `b += x` and `b *= n` on a bytearray run its __iadd__ and __imul__ --
# the object grows in place and the name is bound to it again -- so every
# other name for it sees the change, as CPython's in-place protocol gives;
# and a negative repeat empties it.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect was a NEW object
# bound to the name (`b = b + x`), which compiles and runs; only the alias that
# kept the old bytes, and `is`, show it.

b = bytearray(b"ab")
alias = b
b += b"c"
print(b, alias, b is alias)
b *= 2
print(b, alias, b is alias)
b += bytearray(b"!")
print(alias)
b *= -1
print(b, alias, len(alias), b is alias)


def grow(target: bytearray, extra: bytes) -> bytearray:
    target += extra
    return target


kept = bytearray(b"x")
same = grow(kept, b"yz")
print(kept, same, kept is same)
