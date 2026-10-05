# WHAT: ints and floats a container holds by value -- an int in 63 bits, a
# float whose exponent is in the common range or +0.0 -- read back as the same
# number through every reader: indexing, iteration, equality against an
# object-held value of the same number, hashing, sorting, repr, isinstance on
# an erased read, a closure, and a union container they are copied into. The
# boundary values on either side of each range are the ones held as objects.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which encoding a slot gets
# is a runtime decision on the value, and a wrong decode prints a different
# number (or crashes on an address made of the value's bits). This passes on
# the build before immediates too; it guards the readers, not a defect.
def ints() -> list[int]:
    xs: list[int] = []
    for v in [0, 1, -1, 257, -6, 2 ** 62 - 1, -(2 ** 62), 2 ** 62, -(2 ** 62) - 1, 2 ** 63, 10 ** 30]:
        xs.append(v + 0)
    return xs


def floats() -> list[float]:
    fs: list[float] = []
    for v in [0.0, -0.0, 1.0, -2.5, 2.0 ** -255, 2.0 ** -256, 2.0 ** 255, 2.0 ** 256, 1e300, 5e-324, float("inf"), float("-inf")]:
        fs.append(v * 1.0)
    return fs


xs = ints()
fs = floats()
print(xs)
print(fs)
print([x + 1 for x in xs])
print([f * 2 for f in fs])
print(sorted(xs), sorted(fs))
print(len(set(xs)), len(set(fs)), sum(xs[:6]))
objects = [2 ** 62 - 1, -(2 ** 62), 2 ** 62]
print([o in xs for o in objects], [xs.index(o) for o in objects])
print(2.0 ** -255 in fs, -0.0 in fs, 0.0 in fs, fs.index(-0.0), fs.count(0.0))
print({x: str(x) for x in xs[:6]}[2 ** 62 - 1], hash(xs[5]) == hash(2 ** 62 - 1))
print([repr(f) for f in fs[:6]])
nan = float("nan")
ns: list[float] = []
ns.append(nan * 1.0)
print(ns, ns[0] == ns[0], nan in ns)
u: list[int | float | None] = [None]
for x in xs[:4]:
    u.append(x)
for f in fs[:4]:
    u.append(f)
print(u)


def erased(v: object) -> str:
    if isinstance(v, int):
        return "int " + str(v)
    if isinstance(v, float):
        return "float " + str(v)
    return "other"


print([erased(x) for x in xs[:6]], [erased(f) for f in fs[:4]])


def adder(k: int) -> int:
    def inner() -> int:
        return k + xs[5]
    return inner()


print(adder(xs[1]))
