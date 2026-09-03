# CPython's dataclass and NamedTuple reprs print `__qualname__`, which has no
# module in it: an imported `Point(3, 4)` renders as `Point(x=3, y=4)`. This
# rendered `a_module_of_dataclasses.Point(x=3, y=4)` -- a silent wrong answer
# for every imported dataclass printed, and one a program never notices because
# the module name is a plausible thing to see.
#
# The DEFAULT object repr beside it disagrees on purpose: CPython spells that
# `<module.Class object at 0x...>` WITH the module, which is why this compiler
# was taught to qualify that one.
import a_module_of_dataclasses as m

p = m.Point(3, 4)
print(p, p.norm(), p == m.Point(3, 4))
print(m.Point(1))
print([str(q) for q in [m.Point(1), m.Point(2, 2)]])

pair = m.Pair(1)
print(pair, pair.a, pair.b, pair[0])
