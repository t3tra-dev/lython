# complex had no __hash__, so two equal complex values never found each other
# in a dict and the second lookup raised KeyError. Must run: the point is the
# hash values themselves and the dict lookups they decide.

# CPython's rule: hash(re) + 1000003 * hash(im), wrapping, with -1 mapped to
# -2. A real-valued complex must therefore hash like the int and the float.
print(hash(complex(2, 0)), hash(2), hash(2.0))
print(hash(complex(1, 2)))
print(hash(complex(0.5, 0)), hash(0.5))
print(hash(complex(-1, 0)), hash(-1))

# Equal values reach the same bucket; unequal ones stay apart.
d = {complex(1, 2): "a", complex(3, 4): "b"}
print(len(d), d[complex(1, 2)], d[complex(3, 4)])
d[complex(1, 2)] = "c"
print(len(d), d[complex(1, 2)])

# A set collapses the duplicate.
s = {complex(1, 1), complex(1, 1), complex(2, 2)}
print(len(s))

# Sign of zero: 0.0 and -0.0 hash alike, so complex(0, 0) and complex(0, -0.0)
# are the same key.
z = {complex(0, 0): "zero"}
print(z[complex(0.0, -0.0)])
