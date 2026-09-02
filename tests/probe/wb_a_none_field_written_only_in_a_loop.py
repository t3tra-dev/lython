# FIXED 2026-09-02. Kept as the reproducer.
#
# WAS: a class whose ONLY field is written `= None` inside a loop left the
# constructed object unreleased:
#
#     owned resource from builtin.unrealized_conversion_cast result 0 reaches
#     function exit without release, transfer, or owned return
#
# reported in __main__, at `c = C()` -- reading the field was not needed.
# Everything one step away compiled (field-region sweep, 75 programs): the same
# store at `__init__`'s top level, in both arms of an if/else, or storing a 1 or
# a [1, 2] instead. So it was the pair (NoneType field, written only inside a
# region with a back edge or a handler): `for`, `while` and `try` all failed and
# if/else did not.
#
# ⭐ THE MARKER WAS RE-MINTED AT THE STORE. `markOwnedLocalObjectBundle`
# re-roots the owned-local token at every field store, because a store REPLACES
# lanes and the token has to name the new ones -- and re-minting moved the
# marker into the loop body, where it no longer dominates the function exit the
# release has to be placed at. A field with no runtime lanes replaces nothing,
# so the repair is to keep the marker the allocation already put there when the
# value group is unchanged. Asked by OPERAND IDENTITY rather than by "the field
# has no lanes", which covers `self.xs = self.xs` too.
#
# ⛔ Reachable only since fields assigned inside a region became fields at all
# (2026-09-02); before that this program was "class C has no field 'v'". The
# same widening left a second hole, fixed with it: a field whose value mentions
# the LOOP TARGET (`self.n = i`) took the erased top, because the walk bound the
# region's assignments but not the loop's own target.
class C:
    def __init__(self) -> None:
        for _ in range(1):
            self.v = None


c = C()
print(c.v)
