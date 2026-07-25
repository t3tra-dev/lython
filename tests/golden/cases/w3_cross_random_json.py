# Wave 3 cross-track: the Mersenne Twister port (os-time) drawing through a
# monomorphized generic class (generic-classes) and out through json (text).
#
# What this pins that stdlib_random alone does not: the MT19937 stream is
# still CPython's bit for bit when the draws are consumed by generic-class
# storage and by the json encoder instead of being printed straight. A
# specialization inserts a construction and a __setitem__ between consecutive
# draws, and json's encoder runs a deep call graph between them; either could
# perturb the generator only by drawing from it, so an unchanged sequence is
# the check. Reseeding to 42 and getting the stdlib_random numbers back is the
# same check from the other side.
#
# JSON trees are built with the bulk `arr_of`/`obj_of` (json.py's linear path)
# rather than repeated `set()`/`append()` calls, which are only lowerable in
# the block that defines the node's storage.
import json
import random
from collections import OrderedDict

# --- draws stored into two instantiations of one generic class -------------
random.seed(42)
rolls: OrderedDict[str, int] = OrderedDict()
rolls["a"] = random.randint(1, 100)
rolls["b"] = random.randint(1, 100)
rolls["c"] = random.randint(1, 100)
rolls["d"] = random.randint(1, 100)
print(rolls)

random.seed(42)
reals: OrderedDict[int, float] = OrderedDict()
reals[0] = random.random()
reals[1] = random.random()
reals[2] = random.random()
print(reals)

# The same seed reproduces the same mapping after the generic class has been
# constructed twice already.
random.seed(42)
again: OrderedDict[str, int] = OrderedDict()
again["a"] = random.randint(1, 100)
again["b"] = random.randint(1, 100)
again["c"] = random.randint(1, 100)
again["d"] = random.randint(1, 100)
print(again == rolls)

# --- draws encoded through json -------------------------------------------
random.seed(7)
draw_keys: list[str] = ["x", "y", "z"]
draw_values: list[json.JSONValue] = []
drawn = 0
while drawn < 3:
    draw_values.append(json.of_int(random.randint(0, 999)))
    drawn = drawn + 1
doc = json.obj_of(draw_keys, draw_values)
print(json.dumps(doc))
print(json.dumps(doc, sort_keys=True))

# A draw, an encode, then another draw: the encoder must not touch the stream.
random.seed(7)
first = random.randint(0, 999)
scratch_values: list[json.JSONValue] = [json.of_int(first)]
scratch = json.obj_of(["first"], scratch_values)
print(json.dumps(scratch))
second = random.randint(0, 999)
print(first, second)

# --- shuffle and sample after the same seed, with the results in json -----
random.seed(2026)
deck = [1, 2, 3, 4, 5, 6, 7, 8]
random.shuffle(deck)
print(deck)
picked = random.sample([10, 20, 30, 40, 50, 60], 3)
print(picked)
deck_values: list[json.JSONValue] = []
position = 0
while position < len(deck):
    deck_values.append(json.of_int(deck[position]))
    position = position + 1
print(json.dumps(json.arr_of(deck_values)))

# --- a float stream through json's shortest-repr encoder -------------------
random.seed(99)
float_values: list[json.JSONValue] = []
count = 0
while count < 4:
    float_values.append(json.of_float(random.random()))
    count = count + 1
print(json.dumps(json.arr_of(float_values)))

# --- the ordered mapping decoded back from json ---------------------------
random.seed(11)
bit_values: list[json.JSONValue] = []
bits = 0
while bits < 2:
    bit_values.append(json.of_int(random.getrandbits(16)))
    bits = bits + 1
wire = json.dumps(json.obj_of(["p", "q"], bit_values))
print(wire)
parsed = json.loads(wire)
back: OrderedDict[str, int] = OrderedDict()
back["p"] = parsed.get("p").as_int()
back["q"] = parsed.get("q").as_int()
print(back)
print(back["p"], back["q"])
