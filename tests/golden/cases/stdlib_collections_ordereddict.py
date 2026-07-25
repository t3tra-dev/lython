# collections.OrderedDict: a GENERIC stdlib class from lib/collections.py,
# instantiated at two different key/value pairings in one program. Views are
# spelled list(...) because the port returns materialized lists (documented
# deviation), which keeps this output identical to CPython's.
from collections import OrderedDict

d: OrderedDict[str, int] = OrderedDict()
d["a"] = 1
d["b"] = 2
d["c"] = 3
print(d)
print(len(d), d["b"], "c" in d, "z" in d)
print(list(d.keys()))
print(list(d.values()))
print(list(d.items()))

d.move_to_end("a")
print(list(d.keys()))
d.move_to_end("a", False)
print(list(d.keys()))

print(d.popitem())
print(d.popitem(False))
print(d)

print(d.get("b", 0), d.get("zz", -1))
print(d.setdefault("b", 9), d.setdefault("q", 8))
print(d)

copied = d.copy()
print(copied, copied == d)
copied["r"] = 1
print(copied == d)

merged: OrderedDict[str, int] = OrderedDict()
merged.update(d)
print(merged, merged == d)
merged.clear()
print(merged, len(merged))

# Order-sensitive equality: same items, different insertion order.
left: OrderedDict[str, int] = OrderedDict()
left["x"] = 1
left["y"] = 2
right: OrderedDict[str, int] = OrderedDict()
right["y"] = 2
right["x"] = 1
print(left == right)
right.move_to_end("x", False)
print(left == right)

# A second instantiation of the same class.
numbered: OrderedDict[int, str] = OrderedDict()
numbered[1] = "one"
numbered[2] = "two"
print(numbered)
print(numbered.pop(1))
print(numbered)
del numbered[2]
print(numbered, len(numbered))

try:
    numbered.popitem()
except KeyError as error:
    print("KeyError", error)
