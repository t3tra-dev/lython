# The whitespace split with a cap. `"a b c".split(maxsplit=1)` and its
# `split(None, 1)` spelling were both "builtins.str does not provide manifest
# method 'split'" -- the separator overload takes a maxsplit and the
# whitespace one did not, so the two spellings CPython folds together were a
# working method and a missing one.
print("a b c".split(maxsplit=1))
print("a b c".split(None, 1))
print("a b c".split(maxsplit=0))
print("a b c".split(maxsplit=5))

# The cap engages only when it actually withholds a split: with nothing held
# back there is no remainder, and the remainder is the only part that keeps
# its interior and trailing spaces.
print("  a  ".split(maxsplit=1))
print("a b ".split(maxsplit=1))
print("a  b  c".split(maxsplit=1))
print("".split(maxsplit=1))
print("   ".split(maxsplit=1))

# The spellings that already worked, unchanged.
print("a b c".split())
print("a b c".split(None))
print("a,b,c".split(","))
print("a,b,c".split(",", 1))
print("a,b,c".rsplit(","))

# A computed cap, so the fold is not only for literals.
n = 1
print("x y z".split(maxsplit=n))
print("x y z".split(maxsplit=n + 1))
