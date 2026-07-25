# Wave 3 cross-track: the json port (text) feeding a monomorphized generic
# class (generic-classes), and the generic class feeding json back.
#
# What this pins that neither track pins alone: a JSONValue tree's ints
# survive being stored into three DIFFERENT instantiations of one generic
# class, and a tree rebuilt from that class's iteration order encodes to the
# bytes CPython produces. The two ports live in the same lib/ directory but in
# separate files, so the specialization demanded from json's key type and the
# one demanded from the main module have to agree on one contract.
#
# Why straight-line stores instead of a loop over `parsed.keys()`: an owned
# JSONValue (or an owned key string taken out of a list) that stays live
# across a generic-class method call is rejected by the ownership verifier --
# the unwind landing pad that would release it is the Wave 3 item that did not
# land. The rejection is loud, and this case is deliberately written on the
# accepted side of it; the shapes that are rejected are recorded in the wave
# report rather than pinned here.
import json
from collections import OrderedDict

doc = '{"beta": 2, "alpha": 1, "gamma": 3, "delta": 4}'
parsed = json.loads(doc)

# JSON object order is insertion order on both sides.
counts: OrderedDict[str, int] = OrderedDict()
counts["beta"] = parsed.get("beta").as_int()
counts["alpha"] = parsed.get("alpha").as_int()
counts["gamma"] = parsed.get("gamma").as_int()
counts["delta"] = parsed.get("delta").as_int()
print(counts)
print(len(counts), counts["alpha"], "gamma" in counts, "epsilon" in counts)
print(counts.keys())
print(counts.values())

# The inverse instantiation of the same generic class, keyed by the values.
inverted: OrderedDict[int, str] = OrderedDict()
inverted[counts["beta"]] = "beta"
inverted[counts["alpha"]] = "alpha"
inverted[counts["gamma"]] = "gamma"
inverted[counts["delta"]] = "delta"
print(inverted)
print(inverted[4], inverted[1])

# Rebuild a JSON object from the ordered mapping: the keys come out of the
# generic class in its own order, so sort_keys=False pins that order and
# sort_keys=True pins the encoder's sort. `obj_of` is json.py's linear bulk
# path (repeated `set()` calls rebuild the child list each time).
rebuilt_keys: list[str] = counts.keys()
rebuilt_values: list[json.JSONValue] = [
    json.of_int(counts["beta"]),
    json.of_int(counts["alpha"]),
    json.of_int(counts["gamma"]),
    json.of_int(counts["delta"]),
]
rebuilt = json.obj_of(rebuilt_keys, rebuilt_values)
print(json.dumps(rebuilt))
print(json.dumps(rebuilt, sort_keys=True))
print(json.dumps(rebuilt, indent=2, sort_keys=True))

# A nested document: the array members go through the same accessor chain
# into a third instantiation.
nested = json.loads('{"rows": [{"id": 7, "tag": "x"}, {"id": 9, "tag": "y"}]}')
rows = nested.get("rows")
tags: OrderedDict[int, str] = OrderedDict()
tags[rows.item(0).get("id").as_int()] = rows.item(0).get("tag").as_str()
tags[rows.item(1).get("id").as_int()] = rows.item(1).get("tag").as_str()
print(tags)
tag_keys = tags.keys()
tag_values = tags.values()
print(tag_keys, tag_values)

# A missing key still raises through each port's own accessor: the first
# KeyError comes from json, the second from the generic class's dict.
try:
    parsed.get("epsilon")
except KeyError as error:
    print("KeyError:", str(error))
try:
    print(counts["epsilon"])
except KeyError as error:
    print("KeyError:", str(error))
