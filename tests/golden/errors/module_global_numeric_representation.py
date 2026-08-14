# A module global's cell has one runtime representation, fixed by its
# declaration. An int does not fit a float's, and this used to be reported as
# "assignment value group has 3 values, expected 1" from inside lowering --
# a count, from a pass the author never wrote against.
#
# The rebind rather than the initializer, because the declaration is the part
# that is right: `x = 4` is where the two representations meet.
x: float = 3.0
print(x)
x = 4
print(x)
