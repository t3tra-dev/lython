# Tuple equality/ordering/hashing: element recursion incl. nesting.
print((1, 2) == (1, 2), (1, 2) == (1, 3), (1, 2) != (1, 2))
print((1, 2) < (1, 3), (1, 2) < (1, 2, 0), (2,) > (1, 9), () < (0,))
print((1, 2) <= (1, 2), (1, 3) >= (1, 2))
print(("a", "b") < ("a", "c"), ("b",) > ("a", "z"))
print((1, (2, 3)) == (1, (2, 3)), (1, (2, 3)) < (1, (2, 4)))
print((True, 2) == (1, 2), (1.0, 2) == (1, 2))
print(hash((1, (2, 3))) == hash((1, (2, 3))))
