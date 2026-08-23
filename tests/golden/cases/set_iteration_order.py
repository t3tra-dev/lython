# What this pins: ITERATING a set hands the elements back in its hash table's
# slot order, which is what CPython's set_next does, and not in the order they
# were added. `set_table_order` pins the same thing for repr; the two go
# through different code, because repr is a runtime function and the iteration
# is lowered inline by the compiler, and only one of them was covered.
#
# Why this needs to run: the elements are all present and `len` is right in
# either case. The only observable is the order the walk yields, and there is
# no diagnostic to assert on before execution.
#
# Ints only, and small ones: a small int hashes to itself in both
# implementations, while CPython randomises string hashes per process.

a = set()
a.add(9)
a.add(3)
a.add(6)
a.add(1)
for v in a:
    print(v)
print(list(a))

# A discard leaves a dummy that the next add reuses, so the slot the last
# element lands in is not the one an append would give it.
b = set()
b.add(1)
b.add(9)
b.add(2)
b.discard(1)
b.add(17)
print([v for v in b])

# Enough elements to cross the load factor: the resize re-places every entry,
# so the iteration order after it belongs to the new table.
c = set()
i = 0
while i < 12:
    c.add(i * 5)
    i += 1
print([v for v in c])

# The algebra builds its result by inserting into a fresh table, and a copy
# takes the source's table wholesale; both are iterated here rather than
# printed.
p = set()
p.add(5)
p.add(1)
p.add(9)
p.add(3)
q = set()
q.add(9)
q.add(2)
q.add(5)
print([v for v in p | q], [v for v in p & q], [v for v in p - q])
print([v for v in p.copy()])
print([v for v in frozenset(p)])
