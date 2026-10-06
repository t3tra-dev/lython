from typing import Iterator

# WHAT: ints and floats that containers hold as their value, read back for
# arithmetic alone (where no object is made), unpacked out of tuples built in
# a loop and out of tuples a function returns, expanded into a call with `*`,
# and gathered into sets and dicts -- including ints too wide for an
# immediate, which take the object path.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: whether a read decodes an
# immediate or falls back to an object is decided by each value at run time,
# and a wrong decode prints a different number. Passes on the build before
# these changes too; it guards the readers.


def run(xs: list[int], d: dict[int, int]) -> int:
    t = 0
    best = -1
    for x in xs:
        t += x
        if x > best:
            best = x
    for k in d:
        t = t + d[k] * 2
    return t * 1000 + best


big = 2 ** 70
xs: list[int] = []
for i in range(10):
    xs.append(i * 1000)
d: dict[int, int] = {}
for i in range(5):
    d[i] = i * 3000
print(run(xs, d))
xs.append(big)
print(run(xs, d))
d[99] = -big
print(run(xs, d))
ys: list[int] = []
for i in range(5):
    ys.append(2 ** 62 + i)
print(run(ys, {}))


def add(a: int, b: int) -> int:
    return a + b


def build_pairs(n: int) -> list[tuple[int, int]]:
    xs: list[tuple[int, int]] = []
    for i in range(n):
        xs.append((i * 1000, i * 2000 + 1))
    return xs


ps = build_pairs(4)
print([add(*p) for p in ps])
t = ps[3]
print(add(*t), t, t[0] + t[1])
a, b = ps[2]
print(a, b)
for x, y in ps:
    print(x * y, end=" ")
print()


def pair(i: int) -> tuple[int, float]:
    return (i * 1000, i / 4)


def gen(n: int) -> Iterator[tuple[int, int]]:
    for i in range(n):
        yield (i * 3000, -i * 7000)


def wrap(i: int) -> tuple[tuple[int, int], int]:
    return ((i, i * 1000), 2 ** 70 + i)


qs = [pair(i) for i in range(5)]
print(qs, sorted(qs, reverse=True)[0], max(qs), qs[2] < qs[3], hash(qs[1]) == hash((1000, 0.25)))
a, b = pair(7)
print(a, b, a + 1, b * 2)
s = {pair(i) for i in range(4)}
print(sorted(s), (2000, 0.5) in s)
pd = dict([pair(i) for i in range(3)])
print(pd)
g = list(gen(4))
print(g, [x + y for x, y in g], [f"{x}:{y}" for x, y in gen(3)])
w = [wrap(i) for i in range(3)]
print(w, [inner[1] + big for inner, big in w])
z = list(zip([1000, 2000], [3.5, 4.5]))
print(z, list(enumerate([5000, 6000])))
def keep(t: tuple[int, int]) -> int:
    def inner() -> int:
        return t[0] + t[1]
    return inner()
print(keep((4000, 5000)), keep(g[2]))
o: object = qs[1]
print(o, str(qs[4]), repr(w[2]))
