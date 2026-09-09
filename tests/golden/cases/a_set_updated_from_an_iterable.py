# CPython's set methods take any iterable, and the manifest spelled their
# parameters as a SET -- so the type check refused every one of them:
#
#     a = {1}
#     a.update([2, 3])
#     # builtins.set[int] has manifest method 'update' but no signature
#     # that accepts ...
#
# and the same for `union`, `intersection`, `difference`, their `_update`
# forms, and `issubset`/`issuperset`/`isdisjoint`. `dict.update` had the same
# gap for a sequence of pairs, where CPython takes one and the constructor
# already does (`dict([("b", 2)])` compiles).
#
# The parameters are `Iterable[$T]` now, which is what `list.extend` has always
# said, and the argument is MATERIALIZED into the receiver's own container --
# the runtime implements the peer-container case, so what has to be built is a
# peer. Materializing a list there only moved the refusal to the ABI ("cannot
# adapt builtins.list to runtime input 1 of builtins.set.update").
#
# Why execution: the elements have to actually arrive, and the set operations
# below are only correct if what was built holds what the iterable held.
#
# ⛔ The operators keep the set parameter: `a | [3]` is a TypeError in CPython
# too, and only the named methods take an iterable.
#
# ⛔ And an argument that is already the receiver's own container is left
# alone: `s.update(other_set)` is the shape the runtime implements directly,
# and building a second set around it would be work for nothing.
#
# ⛔ `set()` -- EMPTY -- updated from a str is still refused: the seed scan
# reads what fills a container, and a str's element is a str it cannot tell
# from the str itself. `{"z"}` below has an element already, which is why it
# is spelled that way.

a = {1, 2}
a.update([3, 4])
a.update((5,))
a.update({6})
print(sorted(a))

b = {1, 2, 3}
print(sorted(b.union([4])), sorted(b.intersection([2, 9])), sorted(b.difference([1])))
print(sorted(b.symmetric_difference([3, 7])))
print(b.issubset([1, 2, 3, 4]), b.issuperset([1]), b.isdisjoint([9]))

c = {1, 2, 3}
c.difference_update([1])
c.intersection_update([2, 3, 9])
print(sorted(c))

d = {"a": 1}
d.update({"b": 2})
d.update([("c", 3)])
print(sorted(d.items()))
print(d["c"] + 1)

e = {"z"}
e.update("ab")
print(sorted(e))
