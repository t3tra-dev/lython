# What this pins: a tuple whose members turn out to be the SAME type after a
# generic contract is substituted.
#
#     d = {1: 2}
#     print(sorted(d.items()))
#     # static type list[tuple[int, int]] does not provide manifest method
#     # 'append'
#
# A uniform tuple has one spelling here: `tupleOfMembers` collapses `(1, 2)` to
# the arity-erased `tuple[int]` and keeps `(1, "a")` positional. `dict.items()`
# is declared `list[tuple[$K, $V]]`, and binding K = V = int rebuilt it
# positionally as `tuple[int, int]` -- the spelling the manifest lookup does
# not have. `{"a": 1}.items()` was fine the whole time, because its members
# differ and positional IS the right spelling there, which is why this only
# showed for a dict whose keys and values are the same type.
#
# Why this needs to run rather than assert on a diagnostic: the collapse is a
# TYPE decision with a runtime shape behind it -- an arity-erased tuple and a
# positional one lay out the same members, and unpacking, indexing and
# comparison all have to keep working through the changed spelling. `sorted`
# over the pairs is the sharpest check: it compares them element by element.
#
# Every expected line is python3.14's.

# --- the same type on both sides, which is what broke ---------------------
d = {1: 2, 3: 4}
print(sorted(d.items()))
print(sorted(d.keys()), sorted(d.values()))
print(max(d.items()), min(d.items()))
print(list(d.items()))
for k, v in sorted(d.items()):
    print(k, v)

doubled = {k: v * 2 for k, v in d.items()}
print(sorted(doubled.items()))

names = {"a": "b", "c": "d"}
print(sorted(names.items()))


# --- a dict comprehension whose key and value agree ------------------------
# ⛔ The comprehension is bound to a name first, and that is a boundary rather
# than a style: `.items()` called directly on a comprehension RESULT is
# "runtime manifest has no builtins.dict.items method", before this change and
# after it. A dict LITERAL temporary (`{1: 2}.items()`) is fine, so it is the
# comprehension's result that arrives without the evidence the method needs.
# Recorded in tests/probe/wb_grid_leftovers_2026_08_16.py.
xs = [1, 2, 3]
squares = {x: x * x for x in xs}
print(sorted(squares.items()))
texts = {str(x): str(x) for x in xs}
print(sorted(texts.items()))
print(sorted({1: 2}.items()))


# --- through a class, which is where this was found ------------------------
class Box:
    def __init__(self, v: int) -> None:
        self.v = v

    @property
    def doubled(self) -> "Box":
        return Box(self.v * 2)


boxes = [Box(1), Box(2)]
by_value = {b.v: b.doubled.v for b in boxes}
print(sorted(by_value.items()))


# --- THE CONTROL: members that DIFFER stay positional ----------------------
mixed = {1: "x", 2: "y"}
print(sorted(mixed.items()))
single = {"a": 1}
print(sorted(single.items()))
print(sorted(zip([1, 2], ["a", "b"])))
print(list(enumerate([10, 20])))
