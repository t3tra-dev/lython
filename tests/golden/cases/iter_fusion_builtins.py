# enumerate/zip/map/filter/reversed/iter consumed directly by a for loop
# fuse into rewritten loops; the observable evaluation order must match
# CPython's lazy per-element order (transform/predicate side effects
# interleave with the loop body).
for i, x in enumerate(["a", "b", "c"]):
    print(i, x)
for i, x in enumerate([10, 20], 5):
    print(i, x)
for i, x in enumerate([7, 8], start=2):
    print(i, x)
for p in enumerate(["x", "y"]):
    print(p)
for i, x in enumerate("abc"):
    print(i, x)

xs = [1, 2, 3, 4]
ys = ["p", "q", "r"]
for a, b in zip(xs, ys):
    print(a, b)
for a, b in zip("xyz", [1, 2]):
    print(a, b)
for i, (a, b) in enumerate(zip([1, 2], "ab")):
    print(i, a, b)


def double(v: int) -> int:
    print("double", v)
    return v * 2


for m in map(double, xs):
    print(m)
for m in map(lambda v: v + 100, xs):
    print(m)
for s, n in map(lambda a, b: (a, b * 2), ["k", "l"], [1, 2]):
    print(s, n)


def loud_is_even(v: int) -> bool:
    print("pred", v)
    return v % 2 == 0


for f in filter(loud_is_even, xs):
    print(f)
for w in filter(None, ["", "hey", "", "yo"]):
    print(w)

for r in reversed(ys):
    print(r)
for r in reversed("ab"):
    print(r)
for v in iter(xs):
    print(v)

# break / continue / else through the rewrites
total = 0
for i, x in enumerate([5, 6, 7]):
    if i == 1:
        continue
    total = total + x
print(total)
for i, x in enumerate([5, 6, 7]):
    if x == 6:
        break
    print(i)
else:
    print("no break")
for i, x in enumerate([5, 6, 7]):
    print(i)
else:
    print("exhausted")
