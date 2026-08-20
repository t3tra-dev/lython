# `for i in ...: for j in ...: out.append(i)` was refused with "ownership CFG
# exploration exceeded 20000 states": the append parks a reference to the OUTER
# element inside the INNER cycle, and the walk charged it again on every trip.
# Must run: that the program compiles proves the walk converges, but only the
# printed elements prove the parked references are real ones -- a walk made to
# converge by dropping the charge would compile this and print freed memory.

out: list[int] = []
for i in range(2):
    for j in range(2):
        out.append(i)
print(out)

# The inner element and a constant took the same path before the fix (they were
# already accepted); they must keep working.
inner: list[int] = []
for i in range(2):
    for j in range(3):
        inner.append(j)
print(inner)

# Flattening a matrix is the shape this appears as in real code.
rows = [[1, 2], [3, 4]]
flat: list[int] = []
for r in rows:
    for v in r:
        flat.append(v)
print(flat)

# The same shape as a comprehension with two for clauses, which lowers to the
# same nest.
print([i for i in range(2) for j in range(2)])
print([c for c in "ab" for _ in range(2)])

# Strings, so a wrong lifetime shows up as garbage rather than as a plausible
# small integer.
names = ["alpha", "beta"]
tagged: list[str] = []
for n in names:
    for k in range(2):
        tagged.append(n + str(k))
print(tagged)

# Three deep, with every level's element appended.
deep: list[int] = []
for a in range(2):
    for b in range(2):
        for c in range(2):
            deep.append(a * 100 + b * 10 + c)
print(deep)
