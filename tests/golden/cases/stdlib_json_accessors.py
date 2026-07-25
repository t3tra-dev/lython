"""JSONValue kind predicates, typed accessors and wrong-kind diagnostics.

Companion to stdlib_json_build; split from it because the whole-program
ownership analysis is superlinear in a module's statement count and the two
halves together sat far outside the golden time budget.
"""

import json

# Kind predicates.
shape = '{"n": null, "b": true, "i": 1, "f": 1.5, "s": "x", "a": [], "o": {}}'
probe = json.loads(shape)
print(probe.get("n").is_null())
print(probe.get("b").is_bool())
print(probe.get("i").is_int())
print(probe.get("f").is_float())
print(probe.get("s").is_str())
print(probe.get("a").is_array())
print(probe.get("o").is_object())
print(probe.get("i").is_float())
print(probe.get("f").is_int())

# Typed accessors.
print(probe.get("b").as_bool())
print(probe.get("i").as_int())
print(probe.get("f").as_float())
print(probe.get("s").as_str())
print(len(probe))
print("n" in probe)
print("q" in probe)
print(probe["s"].as_str())

nested = json.loads('[10, [20, 30], {"k": 40}]')
print(nested.item(0).as_int())
print(nested.item(1).item(1).as_int())
print(nested.item(2).get("k").as_int())
print(nested[0].as_int())
print(len(nested))

# A wrong-kind accessor raises TypeError, never a silent coercion.
try:
    probe.get("s").as_int()
except TypeError as e:
    print("TypeError:", str(e))
try:
    probe.get("i").as_str()
except TypeError as e:
    print("TypeError:", str(e))
try:
    probe.get("a").get("k")
except TypeError as e:
    print("TypeError:", str(e))

# A missing object member raises KeyError.
try:
    probe.get("missing")
except KeyError as e:
    print("KeyError:", str(e))
