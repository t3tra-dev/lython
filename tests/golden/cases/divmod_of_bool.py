# divmod() with a bool numerator. Execution is needed because the values are
# the whole assertion: bool inherits int's __divmod__ unchanged in CPython, so
# both halves of the pair come back as ints, and a repair that widened the truth
# bit to the wrong lane would compile and print (1, 0) or (0, 0) just as well.
#
# The int spellings are here so a repair that reroutes them is caught in the
# same file, and the variable form because the constant folds and it does not.

print(divmod(True, 2))
print(divmod(False, 3))
print(divmod(7, 2))

flag: bool = True
print(divmod(flag, 2))
print(divmod(flag, 1))
