# What this pins: an unfrozen dataclass is refused as a dict key or a set
# element, and `@dataclass(frozen=True)` is the spelling that works.
#
#     @dataclass
#     class K:
#         a: int
#     d = {K(1): 2}
#     print(d[K(1)])        # KeyError: K(a=1)
#
# The class was accepted as a key, stored under an identity hash, and then
# MISSED on a key it calls equal. CPython sets __hash__ to None for a class
# that defines __eq__ and inherits object's __hash__ -- every unfrozen
# dataclass -- and this compiler was already refusing `hash(K(1))` for exactly
# that reason. The question simply was not asked at the key.
#
# Why this must run: the answer is which BUCKET a key lands in. Two equal
# instances built separately have to find each other, and a dict of two
# distinct keys has to stay two entries -- neither is visible before the
# program runs, and the identity hash got both wrong in the same direction.
#
# ⛔ frozen=True IS ALSO A REFUSAL: the fields may not be assigned outside the
# constructor, which is what CPython's FrozenInstanceError says. Accepting the
# keyword without that would let a live dict key's hash change under it. The
# refusal is pinned in EmitterTests, at the compile-time boundary CPython
# reaches only at run time.
#
# ⛔ The hash is `hash((self.f0, ...))` -- tuple's own, which is what CPython's
# dataclass builds -- so it agrees with the synthesized __eq__ by construction.
from dataclasses import dataclass


@dataclass(frozen=True)
class Key:
    a: int
    b: str = "z"


@dataclass(frozen=True, order=True)
class Ranked:
    score: int
    name: str


k = Key(1)
print(k, k.a, k.b)
print(Key(1) == Key(1), hash(Key(1)) == hash(Key(1)), Key(1) == Key(2))

d: dict[Key, str] = {}
d[Key(1)] = "one"
d[Key(1)] = "again"
d[Key(2)] = "two"
d[Key(1, "y")] = "other b"
print(len(d), d[Key(1)], d[Key(2)], d[Key(1, "y")])

s = {Key(1), Key(1), Key(2)}
print(len(s), Key(1) in s, Key(9) in s)

rows = [Ranked(2, "b"), Ranked(1, "a"), Ranked(2, "a")]
print(sorted(rows))
print(len({Ranked(1, "a"), Ranked(1, "a")}))

hits = 0
i = 0
while i < 200:
    table: dict[Key, int] = {Key(i % 5): i}
    if Key(i % 5) in table:
        hits += 1
    i += 1
print("hits", hits)
