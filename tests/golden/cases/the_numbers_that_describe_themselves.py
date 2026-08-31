# What: the four constant attributes of the numeric tower. Running it is what
# shows each folded to the right VALUE -- they all have the same type as the
# receiver, so a wrong fold is a wrong number rather than a failure, and a
# bool's are an int's rather than a bool's.
def parts(n: int) -> str:
    return (str(n.real) + "/" + str(n.imag) + "/" + str(n.numerator) + "/" +
            str(n.denominator))


print(parts(5), parts(-4), parts(0))
print((3.5).real, (3.5).imag, (-0.5).real)
print(True.numerator, True.real, True.denominator, False.numerator)
print(sum(x.numerator for x in [1, 2, 3]))
