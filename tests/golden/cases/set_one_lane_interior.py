# builtins.set is one handle (memref<11xi64>) and builtins.frozenset another
# (memref<13xi64>), each with word 4 holding its items array's base address, so
# a reallocation is a write THROUGH the root and every holder observes it. This
# case pins that property in the shapes a lane travelling beside the root could
# have faked. Most growths here cross the initial 64-slot capacity, because
# below it there is no reallocation and so nothing to get wrong -- shape 11 is
# the deliberate exception, and it is a second mechanism rather than a weaker
# instance of the first.
#
# RED ON MAIN `7822be4`, in five places, SILENTLY: exit 0 with wrong answers,
# which is the "never silently mis-execute" violation rather than a refusal this
# conversion relaxes. Shapes 3 and 4 print `0 False ...` where CPython prints
# `152 True True True` etc. because the pre-mutation name keeps the pre-growth
# `items` lane that `ensure_capacity` has freed; under libgmalloc main SIGSEGVs
# on that read in BOTH guard orders, and under `MallocScribble` it prints a
# DIFFERENT wrong answer, which is how a freed read is told apart from a logic
# bug. Under `--release` (`enableVerifiers = !releaseMode`) nothing refuses it.
#
# Set order is an implementation detail in CPython too, so everything is
# printed through sorted() or through a membership probe.

# Shape 1: a member read back after the array moved. A holder keeping the old
# items lane would probe the freed block.
s: set[int] = set()
i = 0
while i < 200:
    s.add(i)
    i = i + 1
print(len(s), 0 in s, 63 in s, 64 in s, 199 in s, 200 in s)

# Shape 2: a second binding to the same set, probed after the FIRST binding
# grew it past capacity. Both names are the same handle.
t = s
s.add(200)
print(len(t), 200 in t, 0 in t)

# Shape 3: update() driving a growth, then a probe through the PRE-update
# binding. update is void now, so the receiver bundle is handed straight back
# and its element evidence has to be demoted rather than replaced.
u: set[int] = {1, 2}
before = u
big: set[int] = set()
k = 0
while k < 150:
    big.add(k + 1000)
    k = k + 1
u.update(big)
print(len(before), 1 in before, 1000 in before, 1149 in before)

# Shape 4: the in-place filters, which compact through the handle rather than
# returning a renamed representation. Read through the pre-mutation name after
# each one.
f1: set[int] = set()
m = 0
while m < 120:
    f1.add(m)
    m = m + 1
f1a = f1
f1.intersection_update({x for x in range(100, 240)})
print(len(f1a), 99 in f1a, 100 in f1a, 119 in f1a)

f2: set[int] = set()
n = 0
while n < 120:
    f2.add(n)
    n = n + 1
f2a = f2
f2.difference_update({x for x in range(0, 100)})
print(len(f2a), 99 in f2a, 100 in f2a, 119 in f2a)

f3: set[int] = set()
p = 0
while p < 120:
    f3.add(p)
    p = p + 1
f3a = f3
f3.symmetric_difference_update({x for x in range(80, 200)})
print(len(f3a), 79 in f3a, 80 in f3a, 150 in f3a, 199 in f3a)

# Shape 5: a mutation AFTER a growth, which is a different question from a read
# after it -- a void mutator hands the receiver back, so evidence naming the
# pre-growth contents answers the read correctly and then mis-executes the next
# statement. That asymmetry is why the reads above are not enough.
s.discard(0)
s.remove(1)
print(len(s), 0 in s, 1 in s, 2 in s, 199 in s)

# Shape 6: a field slot on a call-produced instance, grown through a LOCAL
# alias of the slot and then read back through the slot. The slot holds the
# handle, so the growth does not have to be written back into it -- which is
# the property under test, and it is the field-slot case a cached lane in the
# box would fake.
#
# Written through a local alias because `b.members.add(x)` itself is refused
# ("set.add requires a rebindable local receiver"), identically on main and
# here -- a pre-existing surface limit this conversion does not change. It is
# now obsolete in principle, since add is void and publishes through the
# handle, but retiring it is a lowering change rather than a manifest one.
class Box:
    def __init__(self) -> None:
        self.members: set[int] = set()


def make() -> Box:
    return Box()


b = make()
alias = b.members
q = 0
while q < 130:
    alias.add(q * 2)
    q = q + 1
print(len(b.members), 0 in b.members, 128 in b.members, 129 in b.members)

# Shape 7: the binary algebra, each of which allocates a set of its own and
# fills it by repeated insertion (so it reallocates on the way).
left: set[int] = {x for x in range(150)}
right: set[int] = {x for x in range(100, 250)}
print(len(left | right), len(left & right), len(left - right), len(left ^ right))
print(sorted(left & right)[0], sorted(left & right)[49])
print(len(left.copy()), 149 in left.copy())

# Shape 8: iteration driven to COMPLETION over a set that reallocated while it
# was being filled. Section 8.6: every loop-carried golden in the suite either
# raised inside the loop or did not mutate in it, which is how the dropped
# borrow-edge retain went unseen.
total = 0
for v in left:
    total = total + v
print(total)

# Shape 9: frozenset, at its own width. frozenset(iterable) is polymorphic over
# all four sequence contracts, so it takes a count and an items array rather
# than any one contract's shape; feeding it a 300-element list with repeats
# exercises the dedupe path through a destination that grows.
src: list[int] = []
r = 0
while r < 300:
    src.append(r % 200)
    r = r + 1
fs = frozenset(src)
print(len(fs), 0 in fs, 199 in fs, 200 in fs)

fs2 = frozenset(left)
print(len(fs2), 149 in fs2, 150 in fs2)

# Shape 10: frozenset's own algebra and its commutative hash, both reading the
# items array through the 13-word handle.
fa = frozenset({x for x in range(120)})
fb = frozenset({x for x in range(80, 200)})
print(len(fa | fb), len(fa & fb), len(fa - fb), len(fa ^ fb))
print(fa.issubset(fa | fb), (fa | fb).issuperset(fb), fa.isdisjoint(frozenset([500])))
print(hash(fa) == hash(frozenset({x for x in range(119, -1, -1)})))

fstotal = 0
for w in fa:
    fstotal = fstotal + w
print(fstotal)

# Shape 11: update through an alias WITHOUT crossing capacity. This is a second
# and distinct main-tree wrong answer: shape 3 above needs the reallocation
# (the pre-update name keeps the freed items lane), whereas here the array never
# moves and main still answers 0/False through the pre-update name, because the
# rebind hands `alias2` a renamed triple and leaves the original name behind. On
# main `7822be4` this prints `0 False`; CPython says `121 True`.
one_more: set[int] = {x for x in range(120)}
alias2 = one_more
alias2.add(500)
print(len(one_more), 500 in one_more)
alias2.update({501})
print(len(one_more), 501 in one_more)

# Shape 12: self-aliasing operations, where the source view and the growing
# destination are the same entity.
selfish: set[int] = {x for x in range(120)}
selfish.update(selfish)
selfish.intersection_update(selfish)
print(len(selfish), len(selfish.union(selfish)), len(selfish.difference(selfish)))

# Shape 13: a frozenset as a dict key and as an element of a set, both of which
# box the handle. A box caching a lane count would fake either.
keyed = {fa: "a", fb: "b"}
print(keyed[frozenset({x for x in range(120)})], keyed[fb])
nested: set[frozenset[int]] = {fa, fb, frozenset({x for x in range(120)})}
print(len(nested))
