# An unannotated generator yields a value narrowed out of a LOCAL `str |
# None` -- in a ternary, and in the `else` arm of an `if`. The walk that
# types the generator's elements read the local from its own table, which
# the ternary's None test did not consult, and it gave the `else` arm no
# fact at all; the elements came out as `object` (or the union was asked
# for `__len__`), while a PARAMETER of the same type compiled.
# Why run it: the sum is only right if each arm yields the int it computes.


def by_ternary(flag: bool):
    held: str | None = "bb" if flag else None
    yield 1
    yield 0 if held is None else len(held)


def by_else(flag: bool):
    held = "ccc" if flag else None
    if held is None:
        yield 0
    else:
        yield len(held)


total = 0
for flag in [True, False]:
    for x in by_ternary(flag):
        total += x
    for x in by_else(flag):
        total += x * 10
print(total)
