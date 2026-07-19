d: dict[tuple[int, int], int] = {}
d[(1, 2)] = 3
bad = (1, [2, 3])
print(hash(bad))
