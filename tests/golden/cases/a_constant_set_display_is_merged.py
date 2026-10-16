# Why execution: a set's printed order is its table, and nothing before the
# run decides the table -- only the printed line tells a merged table from
# an added one.
#
# CPython's compiler folds a set display of three or more constants into a
# frozenset and the bytecode merges it into an empty set, so its table is
# set_merge's, not the adds'. Whether it folds is the compiler's question:
# a display with one name in it is built by adds. Iterated by a `for` or a
# comprehension, or searched by `in`, the frozenset is used as it is.
#
# And the merges themselves: `update`, `set(s)`, `frozenset(s)`, `set(fs)`
# and the operators all read the source in TABLE order, which a set filled
# by adds is not in until something reorders it.
print({0, 7, 14, 21, 28})
print({0, 7, 14, 21, 28, 35})
print({-20, -15, -27, -23, -22, 2})
print({-8, 54, 5, 60 + 1, 48, 67, 43, 78, 26 + 1, -37, 19, 59})
print({2 ** 3, 16, 24, 1 << 5, 40})
print({(1, 2), 0, 7, 14, 21})
print({0.5, 0, 7, 14, 21})
print(list({0, 7, 14, 21, 28}))

a = 0
print({a, 7, 14, 21, 28})
for x in {0, 7, 14, 21, 28}:
    print(x, end=" ")
print()
print([x for x in {0, 7, 14, 21, 28}])
print(14 in {0, 7, 14, 21, 28})


def build(xs: list[int]) -> set[int]:
    s: set[int] = set()
    for x in xs:
        s.add(x)
    return s


k = build([67, 5, -37, 59, 43, 78, 48, 19, 54, -8, 27, 61, 1])
merged: set[int] = set()
merged.update(k)
print(merged)
print(set(k))
print(frozenset(k))
print(set(frozenset(k)))
print(k.copy())
print(k | build([3, 99, -5]))
print(k & build([59, 1, 5, 27, -37, 3]))
print(k ^ build([59, 1, 300, 5]))
t = build([90, 3, 61])
t.symmetric_difference_update(k)
print(t)
