"""The JSONValue builder API: factories, incremental building, bulk builders.

This half of the surface is Lython-specific (CPython has no JSONValue), so
the expected stdout was produced by an equivalent CPython script that builds
the same documents out of dicts and lists and dumps them with stdlib json --
the wire format is the shared contract.
"""

import json


# Factories and dumps.
print(json.dumps(json.null()))
print(json.dumps(json.of_bool(True)))
print(json.dumps(json.of_bool(False)))
print(json.dumps(json.of_int(0)))
print(json.dumps(json.of_int(-7)))
print(json.dumps(json.of_int(10**30)))
print(json.dumps(json.of_float(2.5)))
print(json.dumps(json.of_float(-0.0)))
print(json.dumps(json.of_str("a\"b\\c\nd")))
print(json.dumps(json.arr()))
print(json.dumps(json.obj()))

# Incremental array and object building.
a = json.arr()
a.append(json.of_int(1))
a.append(json.of_str("two"))
a.append(json.null())
print(json.dumps(a))

o = json.obj()
o.set("x", json.of_int(1))
o.set("y", json.of_bool(False))
o.set("z", a)
print(json.dumps(o))
print(json.dumps(o, indent=2))
print(json.dumps(o, sort_keys=True))
print(len(o))
print(o.keys())

# set() on an existing key keeps the position and takes the last value.
o.set("x", json.of_int(99))
print(json.dumps(o))
print(len(o))
print(o.keys())
print(o.get("x").as_int())

# Bulk builders, for elements produced by a loop.
items: list[json.JSONValue] = []
i = 0
while i < 4:
    items.append(json.of_int(i * i))
    i = i + 1
print(json.dumps(json.arr_of(items)))

keys: list[str] = ["k0", "k1", "k2"]
values: list[json.JSONValue] = []
j = 0
while j < 3:
    values.append(json.of_str("v" + str(j)))
    j = j + 1
print(json.dumps(json.obj_of(keys, values)))
