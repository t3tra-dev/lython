# dict iteration and the view methods in for-iterable position, over both
# tiers: the literal (evidence-backed) dict iterates its live payload, and
# the runtime-mode dict (crossed a function boundary) uses the same path —
# insertion order, deletions, and the mutation guard behave identically.
d = {"a": 1, "b": 2, "c": 3}
for k in d:
    print(k)
for k in d.keys():
    print(k)
for v in d.values():
    print(v)
for k, v in d.items():
    print(k, v)
del d["b"]
for k, v in d.items():
    print(k, v)
d["z"] = 9
for k in d.keys():
    print(k, d[k])
total = 0
for v in d.values():
    total = total + v
print(total)
for p in d.items():
    print(p)


def make() -> dict[str, int]:
    fresh: dict[str, int] = {}
    fresh["x"] = 10
    fresh["y"] = 20
    return fresh


r = make()
for k, v in r.items():
    print(k, v)

try:
    for k in d:
        d["boom"] = 0
except RuntimeError as e:
    print("RuntimeError:", e)
