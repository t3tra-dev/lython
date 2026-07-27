# builtins.tuple is one handle (memref<14xi64>) whose word 4 holds the items
# array's base address, so the entity has ONE root and a holder cannot keep a
# stale description of it. A tuple never grows, so -- unlike list -- the shapes
# a travelling lane could have faked are not about reallocation. They are about
# a boxed element being read back through the root by a DIFFERENT consumer than
# the one that wrote it: the box16 slot caches every lane of a multi-lane
# entity, so a three-lane tuple could be read out of a cache that a one-lane
# tuple no longer has (playbook 8.5, __ly_unicode_item_words / b"-".join).
#
# Shape 1: elements read back out of a tuple produced by a CALL, so the
# producing frame is gone by the time they are read.
def make(n: int) -> tuple[int, int, int]:
    return (n, n + 1, n + 2)


t = make(10)
print(len(t), t[0], t[1], t[2])
print(t)

# Shape 2: a payload derived from a concatenation and from a slice -- the two
# producers that allocate a fresh items array and copy boxes into it.
a = (1, 2)
b = (3, 4, 5)
c = a + b
print(c, len(c), c[0], c[4])
s = c[1:4]
print(s, len(s), s[0], s[2])
r = a * 3
print(r, len(r), r[5])

# Shape 3: the MIXED numeric tower through tuple slots. This is the shape that
# segfaulted when float went to one lane (__ly_boxed_float_value read box word
# 5, float's old lane-1 address) and it is the shape that reaches
# __ly_boxed_long_view, which still indexes int's lanes 1 and 2 by position.
# Equality, ordering, hashing and the float coercion all route through it.
print((1.0,) == (1,), (1,) == (1.0,), (True,) == (1,))
print((1, 2.0, True) == (1.0, 2, 1))
print((1.0,) == (2.0,), 1.0 == 1)
print((1, 2) < (1.0, 3), (1.5,) < (2,), (True, 2) < (1, 3.0))
print(hash((1,)) == hash((1.0,)))
print(sorted([(2.0,), (True,), (1,)]))

# Shape 4: a tuple as a dict key, read back after the mapping rehashed past its
# initial capacity. The key's hash and equality both go through the handle.
d: dict[tuple[int, int], int] = {}
i = 0
while i < 200:
    d[(i, i + 1)] = i
    i = i + 1
print(len(d), d[(0, 1)], d[(63, 64)], d[(64, 65)], d[(199, 200)])

# Shape 5: a tuple as a set member, after the set grew past its initial
# capacity, plus a membership probe with a numerically-equal-but-differently-
# typed element.
st: set[tuple[int, int]] = set()
j = 0
while j < 200:
    st.add((j, j + 1))
    j = j + 1
print(len(st), (0, 1) in st, (199, 200) in st, (500, 501) in st)

# Shape 6: a nested tuple -- the inner handle is a boxed element of the outer
# one, so this reads a one-lane handle out of a box slot.
n = ((1, 2), (3, 4))
print(n, n[0], n[1], n[0][1], len(n[0]))
print(n == ((1, 2), (3, 4)), n == ((1, 2), (3, 5)))

# Shape 7: the tuple-producing boundaries owned by OTHER contracts, which the
# step-4 gate named: divmod (builtins.tuple primitive), str.partition and
# str.rpartition (builtins.str methods), exception .args (BaseException), and
# tuple(xs) (a builtins.list primitive).
print(divmod(7, 2), divmod(-7, 2))
print("a-b-c".partition("-"), "a-b-c".rpartition("-"))
print("nosep".partition("-"))
print(tuple([9, 8, 7]))
try:
    raise ValueError("boom")
except ValueError as e:
    # Deliberately three statements rather than `print(e.args, len(e.args),
    # e.args[0])`. That one-line form renders e.args[0] as `'boom'` instead of
    # `boom`. The axis is arity, not position: `print(1, e.args[0])` is wrong
    # too, and `print(e.args[0])` is right. Multi-argument print stringifies
    # each argument statically and admits __str__ only for the exception
    # taxonomy, so an object-typed slot falls to __repr__; single-argument print
    # dispatches through the manifest and reaches object.__str__. Widening that
    # gate to admit object is measured WRONG -- it turns 7 of 13 payload classes
    # into compile errors, because the erased __str__ call is then specialized
    # to the payload contract and list/dict/set/frozenset/tuple/range/NoneType
    # have no __str__ to specialize to. The repair is to stop specializing, not
    # to widen the gate. It reproduces IDENTICALLY on main with tuple at three
    # lanes, so it is pre-existing and not this conversion's; it is left out
    # rather than pinned, because a golden case that encodes it would make the
    # bug a requirement.
    print(e.args)
    print(len(e.args))
    print(e.args[0])

# Shape 8: count/index/contains, the three box primitives, over a mixed tuple.
m = (1, 2.0, True, 2.0)
print(m.count(2.0), m.index(2.0), 2.0 in m, 9 in m)
print(repr(m))
