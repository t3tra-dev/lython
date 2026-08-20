# What this pins: `*` inside a list/set/tuple display and `**` inside a dict
# display.
#
#     print([*xs, 4])       # unsupported expression kind 'Starred'
#     print({*xs, 4})       # starred elements in a set literal are not
#                           # supported yet
#     print({**d, "b": 2})  # dict literal pack has an odd operand count
#
# The third is the one that names the shape of the fix: a literal is PACKED,
# and a pack takes a count the compiler knows. A star does not have one, so the
# literal is not a pack -- it is the loop it means, built left to right out of
# pieces that already compile (an empty accumulator, `for e in xs`, append/add,
# a key store, and `tuple(list)` for the frozen form).
#
# Why this must run: order and duplicates are the answer. `{**d, "a": 9}` keeps
# d's position for "a" and takes the later value, `[*xs, *ys]` is a
# concatenation rather than a set union, and a set literal folds the repeats --
# none of which a type can show. The loop at the end builds all four shapes 1000
# times; tests/leak_gate.py reads 0 for this file.
#
# ⛔ THE ELEMENT TYPE IS COMPUTED UP FRONT, not left to the appends. The seed is
# an empty accumulator and an empty accumulator has no element type of its own,
# so `[*ints, "a"]` would seed list[int] and then refuse the str. The join over
# every piece -- each starred operand's ELEMENT type, each plain element's own
# -- is what makes the mixed case type the way the same literal without a star
# does.
#
# ⛔ And the seed is an empty PACK of that type rather than a synthesized
# `set()` call: the expectation does not reach a construction, so `set()` came
# back set[object] and the first `add` had nowhere to put an int.
xs = [1, 2]
ys = [3, 4]
words = ["a", "b"]
d = {"a": 1, "b": 2}

print([*xs], [0, *xs, 5], [*xs, *ys], [*range(3)], [*"ab"])
print([*xs, "a"], [*words, *xs])
print({*xs}, {*xs, 5}, {*xs, *xs} == {*xs})
print((*xs,), (0, *xs, 5), (*xs, *ys))
print({**d}, {**d, "c": 3}, {**d, "a": 9}, {"z": 0, **d})

empty: list[int] = []
print([*empty], [*empty, 1], len({*empty, 1}))


def pair() -> list[int]:
    return [7, 8]


print([*pair(), 9])

n = 0
i = 0
while i < 300:
    row = [i, i + 1]
    seen = {**{"a": i}, "b": i}
    n += len([0, *row, 3]) + len(seen) + len({*row, i}) + len((*row, i))
    i += 1
print("loop", n)
