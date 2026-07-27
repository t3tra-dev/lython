# One-lane scalar entities (float, complex, range) carried through loops that
# RUN TO COMPLETION, plus the same entities passed as borrowed parameters and
# returned through a loop merge.
#
# What this pins that the other scalar goldens do not. A borrow-edge retain at a
# block-arg merge is spelled by narrowing the entity handle to the two refcount
# words. While every builtins contract's handle was memref<2xi64> that narrowing
# was a memref.cast; a one-lane handle is wider (float 3 words, complex 4, range
# 5) and a cast cannot shrink a static extent, so the lend can be dropped. A
# dropped lend is invisible to the affine verifier -- it belongs to the argument
# reconciling two groups, so each group's own retain/release arithmetic still
# balances -- and it is only observable once the loop reaches its release, i.e.
# once it completes. Nested loops are load-bearing for the same reason the dict
# case needs three links: the inner header is a second merge of the same entity.
#
# Guard-rail, not a feature test: every value below is CPython 3.14's.

v = 0.0
for i in range(3):
    for j in range(3):
        v = v + 1.0
print(v)


def accumulate() -> float:
    total = 0.0
    for a in range(4):
        for b in range(2):
            total = total + 0.5
    return total


print(accumulate())


# A borrowed one-lane parameter returned through a loop merge, on the path where
# the loop body never runs (so the returned value IS the caller's entity) and on
# the path where it does.
def scale(x: float, n: int) -> float:
    while n > 0:
        x = x * 2.0
        n = n - 1
    return x


print(scale(1.5, 0), scale(1.5, 3))


def rotate(z: complex, n: int) -> complex:
    while n > 0:
        z = z * (0.0 + 1.0j)
        n = n - 1
    return z


print(rotate(1.0 + 0.0j, 0), rotate(1.0 + 0.0j, 2))


def narrow(r: range, n: int) -> range:
    while n > 0:
        r = range(n)
        n = n - 1
    return r


print(len(narrow(range(7), 0)), len(narrow(range(7), 3)))

# A range iterator advanced to exhaustion twice over the same range object: the
# iterator is the loop-carried entity and its state word is written through the
# handle, so the second pass must see a fresh iterator rather than the first
# pass's exhausted one.
r2 = range(4)
s1 = 0
for x1 in r2:
    s1 = s1 + x1
s2 = 0
for x2 in r2:
    s2 = s2 + x2
print(s1, s2, len(r2))
