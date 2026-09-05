# What: a union whose active member is a bool, stored in a container and read
# back. `builtins.bool` is the one contract whose value is a truth bit with no
# header, and a slot needs one; the store normalizes through the contract's
# `box` primitive and the read undoes it. Runtime values, because the question
# is which member each element decodes as -- RETURNING the same union has always
# worked, and so has a plain `[True, False]`, which is what said the union's
# spliced member storage was the gap.


def classify(n: int):
    if n < 0:
        return False
    return n * 2


xs = [classify(-1), classify(1)]
print(xs)
print(xs[0], xs[1])

pair = (classify(-2), classify(3))
print(pair, pair[0], pair[1])

table = {"neg": classify(-3), "pos": classify(4)}
print(sorted(table.items()))
print(table["neg"], table["pos"])

rows = [[classify(-4)], [classify(5)]]
print(rows)
print(rows[0][0], rows[1][0])

print([classify(v) for v in (-1, 1, -2, 3)])
