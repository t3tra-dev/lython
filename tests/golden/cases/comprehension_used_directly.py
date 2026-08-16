# What this pins: a comprehension used WITHOUT being bound to a name first.
#
#     print(sorted({x: x * x for x in xs}.items()))
#     # runtime manifest has no builtins.dict.items method
#
# `keys()`, `values()` and `items()` have no runtime object -- they are sugar
# the emitter answers by iterating the dict -- and the sugar asks whether the
# receiver is a dict. The type walk had no arm for a comprehension at all, so
# it answered `builtins.object` and the sugar declined. Binding the result to a
# name first worked, and so did the same call on a dict LITERAL temporary; the
# receiver's TYPE is the whole difference. The value was always built
# correctly, nothing could say what it was.
#
# Why this needs to run rather than assert on a diagnostic: what the arm
# decides is the ELEMENT type as well as the container -- `{x: str(x) ...}` is
# `dict[int, str]` and `{str(x): x ...}` is `dict[str, int]`, and the two
# print differently sorted. A comprehension inferred at the wrong element type
# compiles and reorders the output.
#
# A TUPLE target -- `for i, n in rows` -- is distributed the way the generator
# walk distributes one: positionally from a positional tuple, uniformly from a
# one-argument container. That is most of the comprehensions written over a
# list of pairs.
#
# ⛔ Every part must infer to something concrete. A chained comprehension's
# second `iter` may mention the first target, which the arm does not bind, so
# it falls back to `object` -- the answer it gave before -- rather than
# guessing.
#
# Every expected line is python3.14's.

xs = [1, 2, 3]
words = ["aa", "b", "ccc"]

# --- the three dict views on a comprehension result ------------------------
print(sorted({x: x * x for x in xs}.items()))
print(sorted({x: str(x) for x in xs}.values()))
print(sorted({str(x): x for x in xs}.keys()))
print(len({x: x for x in xs}))
print({x: x for x in xs}.get(1))
print(sorted({w: len(w) for w in words}.items()))

# --- list and set comprehensions used directly -----------------------------
print(sorted([x * 2 for x in xs]))
print(sorted({x % 2 for x in xs}))
print([x for x in xs].count(2))
print(max([len(w) for w in words]))
print(sorted([w.upper() for w in words]))

# --- a TUPLE target, over a list of pairs ---------------------------------
rows = [(1, "a"), (2, "b"), (3, "c")]
print(sorted([i for i, _ in rows], reverse=True))
print(sorted([n for _, n in rows]))
print(sorted({i: n for i, n in rows}.items()))
print(sorted({n: i for i, n in rows}.keys()))
pairs = {"a": 1, "b": 2}
print(sorted([k for k, v in pairs.items() if v > 1]))
print(sorted({v: k for k, v in pairs.items()}.items()))


# --- nested, where the inner one is the VALUE ------------------------------
ys = ["a", "b"]
print(sorted([(x, y) for x in xs for y in ys]))
print(sorted({x: [y for y in ys] for x in xs}.items()))


# --- inside a function -----------------------------------------------------
def counts() -> int:
    return len({w: len(w) for w in words}.keys())


print(counts())


# --- THE CONTROL: bound first, and a dict literal temporary ---------------
bound = {x: x * x for x in xs}
print(sorted(bound.items()))
print(sorted({1: 2}.items()))
