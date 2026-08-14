# int.bit_length(). Execution is needed because the implementation was already
# there -- the helper pow, division and _random count with -- and this only put
# it on the manifest surface, so what has to be checked is that the surface
# reaches THAT helper with the right operands: a wrapper that forwarded the raw
# meta instead of the magnitude view compiles and answers off by a digit.
#
# The powers of two either side of a digit boundary are the cases that separate
# a correct width from an approximate one, and the negative is CPython's rule
# that the sign is dropped rather than counted.

n: int = 5
print(n.bit_length())

print((0).bit_length())
print((1).bit_length())
print((2).bit_length())
print((255).bit_length())
print((256).bit_length())
print((-5).bit_length())
print((-1).bit_length())

# Past one 30-bit limb, so the count comes from the top digit and the limbs
# below it rather than from a single word.
print((10**20).bit_length())
print((1 << 61).bit_length())

# bool reaches it through int, the way every other inherited method does.
flag: bool = True
print(flag.bit_length(), False.bit_length())
