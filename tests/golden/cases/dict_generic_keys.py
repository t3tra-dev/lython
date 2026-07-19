# Hash-based dict probes: non-str keys, insertion order, replacement,
# numeric-tower key unification, tuple keys, and the method surface.
d: dict[int, str] = {}
for i in range(5):
    d[i * 3] = "v" + str(i)
print(d)
print(d[6], d.get(9, "?"), d.get(100, "?"))
d[6] = "replaced"
print(d[6], len(d))
print(6 in d, 7 in d)
print(d.pop(0), len(d), d.pop(0, "gone"))
db: dict[int, str] = {}
db[True] = "t"
db[1] = "one"
print(db, len(db))
df: dict[float, int] = {}
df[0.5] = 1
df[2.0] = 2
print(df[0.5], 2 in df, df)
dt: dict[tuple[int, int], int] = {}
dt[(1, 2)] = 12
dt[(3, 4)] = 34
dt[(1, 2)] = 99
print(dt[(1, 2)], len(dt), (3, 4) in dt, (9, 9) in dt)
a: dict[str, int] = {}
a["x"] = 1
a["y"] = 2
b: dict[str, int] = {}
b["y"] = 20
b["z"] = 30
a.update(b)
print(a)
m = a | b
print(m, len(m))
c = a.copy()
c["w"] = 0
print(len(a), len(c), a == c, a == a.copy())
c.clear()
print(c, len(c))
for k in dt:
    print(k)
