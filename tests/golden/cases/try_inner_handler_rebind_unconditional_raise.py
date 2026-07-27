# The loud face of the same lost rebind. Making the inner try's raise
# UNCONDITIONAL turns the silent wrong answer into a refusal ("released owned
# resource from @LyUnicode_FromBytes is used after release"): with only one way
# out of the inner try, the pre-try lane value is released by the handler's rebind
# and forwarded by the enclosing handler on the same path.
#
# Kept as a separate case from the silent one because the two faces are separated
# by exactly one `if`, and a repair that only restores the value would leave this
# one rejected -- the family would look half fixed with every test green.
#
# Why this needs execution rather than a DriverTests success assertion: what is
# being asserted is that the value which used to be double-released is the value
# printed, and a compile cannot say that.
def literal(n: int) -> str:
    d = "0"
    i = 0
    while i < n:
        try:
            try:
                raise KeyError("k")
            except KeyError:
                d = "set"
                raise ValueError("w")
        except ValueError:
            pass
        i += 1
    return d


def computed(n: int) -> str:
    d = "0"
    i = 0
    while i < n:
        try:
            try:
                raise KeyError("k" + str(i))
            except KeyError as inner:
                d = str(inner)[-3:]
                raise ValueError("w")
        except ValueError:
            pass
        i += 1
    return d


print(literal(0))
print(literal(1))
print(literal(200))
print(computed(1))
print(computed(3))
print(computed(200))
