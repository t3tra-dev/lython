# A dict written through a key no constant names. The store has to land on the
# key the expression produced, the entries already there have to survive it,
# and a delete through the same shape has to remove the right one.


def name(prefix: str, n: int) -> str:
    return prefix + str(n)


counts = {"a1": 10, "b2": 20}
counts[name("a", 1)] = 11
counts[name("c", 3)] = 30
print(sorted(counts.items()))
print(counts[name("a", 1)], counts[name("c", 3)])

del counts[name("b", 2)]
print(sorted(counts.items()), len(counts))

totals = {"seen": 0}
totals[name("seen", 0)[:4]] = totals["seen"] + 5
print(totals)
