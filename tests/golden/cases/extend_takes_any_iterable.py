# What this pins: a manifest method that declares `Iterable` taking one.
#
#     xs: list[int] = []
#     xs.extend((1, 2))
#     # cannot adapt builtins.tuple to runtime input 1 of builtins.list.extend
#     #   [values: 'memref<14xi64>', expected 'memref<9xi64>']
#
# and the same for a range, for a str, and for every iterable that is not a list.
# The manifest declares the parameter as the PROTOCOL `Iterable`, so the type
# check accepts them all; the runtime implements the list case, so the refusal
# arrives from the ABI with a memref width in it. `xs.extend(list(t))` compiles,
# which is the whole repair: the callee consumes the argument entirely, so
# materializing it is exact rather than a change of semantics.
#
# It is the same rewrite a generator argument already took ("every manifest method
# that takes an iterable consumes the whole of it"), asked by the DECLARED
# PARAMETER instead of by the argument's contract name.
#
# Why this needs to run rather than assert on a diagnostic: the rewrite decides
# WHEN and HOW MANY TIMES the argument is walked. A range materialized twice, or a
# str walked as bytes, prints a different list -- so every section below prints
# the result, and the counter section says the source was consumed once.
#
# ⛔ An argument of the RECEIVER's own contract is left alone: `s.update(other)`
# on two sets and `xs.extend(other)` on two lists are the shapes the runtime
# implements directly, and materializing those would take a working call and break
# it. Both are sections below.
#
# ⛔ `s.update((1, 2))` -- a set updated from a tuple -- is still refused. The
# rewrite makes it a list, and set.update's runtime wants a set; it was refused
# before too, so nothing regressed, and it needs the manifest to implement the
# other cases rather than a different rewrite here.
#
# Every expected line is python3.14's.

# --- the iterables that were refused --------------------------------------
xs: list[int] = []
xs.extend((1, 2))
print(xs)
xs.extend(range(3))
print(xs)

cs: list[str] = []
cs.extend("ab")
print(cs)
cs.extend(("c", "d"))
print(cs, len(cs))

# --- join, whose parameter is the same protocol ---------------------------
print("-".join(("a", "b")))
print("".join(["x", "y"]))

# --- the source is walked once --------------------------------------------
calls = 0


def source() -> tuple[int, int]:
    global calls
    calls += 1
    return (7, 8)


ys: list[int] = []
ys.extend(source())
print(ys, calls)


# --- THE CONTROLS: the receiver's own contract, untouched -----------------
a: list[int] = [0]
b: list[int] = [1, 2]
a.extend(b)
print(a, b)

s: set[int] = {1}
s.update({2, 3})
print(sorted(s))

d: dict[str, int] = {"a": 1}
d.update({"b": 2})
print(sorted(d.items()))

# --- and a generator, which took this rewrite first -----------------------
def gen():
    yield 5
    yield 6


zs: list[int] = []
zs.extend(gen())
print(zs)
