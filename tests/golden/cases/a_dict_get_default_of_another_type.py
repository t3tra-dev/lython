# WHAT: `d.get(key, default)` with a default that is not the dict's value
# type answers the default itself on a miss and the value on a hit, as
# typeshed's `V | T` overload says: a str, a float, None or an empty list for
# a dict of ints or lists. The receiver, the key and the default are each
# evaluated once and in that order, hit or miss, and an empty `{}` default of
# a dict-of-dicts is a dict the next `.get` can read.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect was a refusal
# at the lowering, so the claim is the values each spelling produces and the
# order the three operands ran in, which only execution shows.

print({1: 2}.get(3, "no"))
m1: dict[int, int] = {1: 2}
v1 = m1.get(3, "no")
print(v1)
m2: dict[int, int] = {1: 2}
print(m2.get(1, 2.5))
m3: dict[str, list[int]] = {}
print(m3.get("b", None), m3.get("b", []))
m4: dict[int, int] = {1: 2}
v4 = m4.get(3, "no")
if isinstance(v4, str):
    print(v4.upper())
def recv() -> dict[str, int]:
    print("receiver")
    return {"a": 1}
def key(k: str) -> str:
    print("key", k)
    return k
def dflt() -> str:
    print("default")
    return "none"
print(recv().get(key("a"), dflt()))
print(recv().get(key("z"), dflt()))
nested: dict[str, dict[str, int]] = {"x": {"y": 2}}
print(nested.get("x", {}).get("y", 0), nested.get("q", {}).get("y", -1))
counts: dict[str, int] = {"a": 3}
labels = [counts.get(w, "missing") for w in ["a", "b"]]
print(labels)
got = counts.get("zz", 0.5)
print(got, counts.get("a", 0.5) + 1)
class Box:
    def __init__(self) -> None:
        self.m: dict[int, str] = {1: "one"}
b = Box()
print(b.m.get(2, None), b.m.get(1, 0))
total = 0
for w in ["a", "b"]:
    v = counts.get(w, None)
    if v is not None:
        total += v
print(total)
for i in range(200):
    print({i: [i]}.get(i + 1, "x"), end="")
print()
