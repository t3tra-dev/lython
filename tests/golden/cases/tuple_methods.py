# tuple concatenation, repetition, count and index: the same box payload as
# list, stamped with the tuple class id so repr and release stay tuple-shaped.
p = (3, 1, 2, 1)
q = (9, 8)
print(p + q)
print(p * 2, q * 0, p * -1)
print(p.count(1), p.count(3), p.count(7))
print(p.index(1), p.index(2))
print(len(p + q), (p + q)[4])

words = ("pear", "fig")
print(words + ("apple",))
print(words * 2)
print(words.count("fig"), words.index("fig"))

# A concatenated tuple is a tuple, not a list: repr keeps the parentheses and
# it stays hashable and comparable.
joined = p + q
print(joined == (3, 1, 2, 1, 9, 8), joined < (4,))
print(hash(joined) == hash((3, 1, 2, 1, 9, 8)))
print(2 in joined, 42 in joined)

nested = ((1, 2), (3,))
print(nested + ((4, 5),))
print(nested.count((3,)), nested.index((3,)))
