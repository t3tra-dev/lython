# `len(obj, /)` is positional-only in CPython, which raises TypeError. This
# used to compile: the fast path read args[0] and never looked at the keyword,
# so `len([1], bogus=2)` printed 1.
print(len([1, 2], bogus=2))
