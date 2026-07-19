# Set algebra over the hash probes (printed sorted: set order is an
# implementation detail in CPython too).
s = {x % 5 for x in range(20)}
t = {x % 3 for x in range(9)}
print(sorted(s), sorted(t))
print(sorted(s.union(t)), sorted(s.intersection(t)))
print(sorted(s.difference(t)), sorted(s.symmetric_difference(t)))
print(sorted(s | t), sorted(s & t), sorted(s - t), sorted(s ^ t))
far = {x for x in range(77, 79)}
print(s.issubset(s | t), (s | t).issuperset(t), s.isdisjoint(far))
same = {x for x in range(5)}
small = {x for x in range(2)}
tiny = {x for x in range(1)}
print(s == same, s != t, small < s, s <= s, s > tiny, s >= s)
v = s.copy()
v.add(99)
v.discard(0)
v.discard(1000)
print(sorted(v), sorted(s))
v.remove(99)
print(sorted(v))
v.clear()
print(len(v), sorted(v))
st = {(x % 2, x % 3) for x in range(12)}
print(len(st), (1, 2) in st, (2, 1) in st)
fs = {x / 2 for x in range(4)}
print(sorted(fs), 1.5 in fs, 1 in fs)
u1 = {x for x in range(5)}
u1.update({x for x in range(3, 8)})
print(sorted(u1))
u2 = {x for x in range(6)}
u2.intersection_update({x for x in range(2, 9)})
print(sorted(u2))
u3 = {x for x in range(6)}
u3.difference_update({x for x in range(2, 4)})
print(sorted(u3))
u4 = {x for x in range(5)}
u4.symmetric_difference_update({x for x in range(3, 8)})
print(sorted(u4))
