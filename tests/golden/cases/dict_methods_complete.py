d = {"a": 1, "b": 2}
v = d.get("a")
if v is None:
    print("none")
else:
    print(v)
w = d.get("zz")
if w is None:
    print("none")
else:
    print(w)
print(d.setdefault("a", 99))
print(d.setdefault("c", 3))
print(d["c"], len(d))
p = d.popitem()
print(p)
print(len(d))
print(d.popitem())
print(d.popitem())
try:
    d.popitem()
except KeyError as e:
    print("KeyError:", e)
f = dict.fromkeys(["x", "y"], 0)
print(len(f), f["x"], f["y"])
g = {"m": 1}
g |= {"n": 2}
print(len(g), g["n"])
h = {"k": 5}
print(len(h.keys()), len(h.values()), len(h.items()))
print("k" in h.keys())
print("z" in h.keys())
print(5 in h.values())
print(6 in h.values())
print(("k", 5) in h.items())
print(("k", 6) in h.items())
print(("z", 5) not in h.items())

# conditional mutation demotes evidence to the runtime tier (silent
# mis-execution guard: stale evidence must not answer reads)
cd = {"a": 1}
if "c" not in cd:
    cd["c"] = 3
print(cd["c"], len(cd))
cd.clear()
try:
    print(cd["a"])
except KeyError:
    print("cleared")
d2get = {"k": 1}
print(d2get.get("k", 0), d2get.get("zz", -7))
print(d2get.pop("k"))
print(d2get.pop("k", -8))
u = {"m": 1}
u.update({"n": 2})
print(len(u), u["n"])
sd: dict[str, int] = {}
print(sd.setdefault("x", 4))
print(sd.setdefault("x", 5))
