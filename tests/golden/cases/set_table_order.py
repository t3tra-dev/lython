# What this pins: a set comes out in its HASH TABLE's slot order, which is
# what CPython prints, and not in the order the elements were inserted.
#
# Why this needs to run rather than assert on a diagnostic: the whole defect
# was a stdout divergence with no diagnostic anywhere -- the elements were all
# present and `in` and `len` were right, only the order the walk handed them
# out in was wrong. There is nothing to assert on before execution.
#
# Every expected line below is python3.14's. Elements are ints only, and
# small ones: a small int hashes to itself in both implementations, while
# CPython randomises string hashes per process, so a set of strings has no
# stable answer to pin (measured: three runs of `print({"b","a","c","d"})`
# gave three different orders).
#
# The shapes are chosen so each exercises one thing the order depends on:
#   - insertion order that is NOT slot order (the defect itself)
#   - a collision, so the linear-probe run is what places the second element
#   - a discard, so a DUMMY is what the next insert reuses
#   - enough elements to cross the load factor and force a resize, which
#     re-places everything in the new table's order
#   - the algebra, whose results are built by inserting into a fresh table

# --- insertion order is not the answer -------------------------------------
print({1, 0})
print({2, 1, 0})

a = set()
a.add(9)
a.add(3)
a.add(6)
a.add(1)
print(a)

# --- a collision: 8 and 0 both want slot 0 in an 8-slot table --------------
b = set()
b.add(8)
b.add(0)
print(b)

c = set()
c.add(0)
c.add(8)
print(c)

# --- a dummy is reused, so the discard is visible in the ORDER -------------
d = set()
d.add(1)
d.add(9)
d.add(2)
d.discard(1)
d.add(17)
print(d)

# --- crossing the load factor re-places every entry ------------------------
e = set()
i = 0
while i < 12:
    e.add(i * 5)
    i += 1
print(e)
print(len(e))

# --- the algebra builds its result through the same table ------------------
p = set()
p.add(5)
p.add(1)
p.add(9)
p.add(3)
q = set()
q.add(9)
q.add(2)
q.add(5)
print(p | q)
print(p & q)
print(p - q)
print(p ^ q)
print(p.copy())

r = p.copy()
r |= q
print(r)
r = p.copy()
r &= q
print(r)
r = p.copy()
r -= q
print(r)
r = p.copy()
r ^= q
print(r)

# --- membership and length are unaffected by any of it ---------------------
print(0 in {1, 0}, 7 in {1, 0}, len({2, 1, 0}))
print(frozenset(p) == frozenset(p.copy()))
