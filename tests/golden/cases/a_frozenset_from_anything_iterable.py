# What: a frozenset has no literal spelling, so the only way to see that the
# constructor took the argument's ELEMENTS -- and dropped its duplicates -- is
# to build one from each kind of source and print what came out.
print(sorted(frozenset("abca")))
print(sorted(frozenset(range(3))))
print(sorted(frozenset((1, 2, 2))))
print(sorted(frozenset([3, 1, 3])))
print(sorted(frozenset({4, 5})))
print(sorted(frozenset({"a": 1, "b": 2})))
print(sorted(frozenset(frozenset([7, 7, 8]))))
print(sorted(frozenset(b"aab")))

letters = frozenset("hello")
print(len(letters), "h" in letters, "z" in letters)
print(sorted(letters | frozenset("world")))
print(sorted(letters & frozenset("world")))


def unique(text: str) -> "frozenset[str]":
    return frozenset(text)


print(sorted(unique("mississippi")))
