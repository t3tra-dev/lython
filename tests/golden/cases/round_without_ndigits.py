# int.__round__() with no ndigits. Execution is needed because the repair is a
# CONTRACT entry, not code: the implementation already defaulted ndigits through
# `ly.runtime.default_i64`, so a wrong entry compiles and rounds to the wrong
# power of ten rather than failing. Only the printed values separate "ndigits
# defaulted to 0" from "ndigits picked up whatever was in the slot".
#
# round(2.5) is here for the tie: CPython rounds half to even, so 2 and not 3,
# and an implementation swapped for a naive one would still pass every other
# line in this file.

n: int = 5
print(n.__round__())
print(n.__round__(0))
print(round(n))

print(round(True), round(False))
flag: bool = True
print(round(flag), flag.__round__())

print(round(2.6), round(2.5), round(3.5), round(-3.7))
print(round(1234, -2), round(1255, -1))
