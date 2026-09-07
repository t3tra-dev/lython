# A union FIELD read straight into `print` or `str` is the half of
# wb_union_field_two_owning_members that the dead-value tag did not fix.
#
#   class Box:
#       def __init__(self, v: "int | str") -> None:
#           self.v: "int | str" = v
#   b = Box(7)
#   print(b.v)      # owned resource from builtin.unrealized_conversion_cast
#                   # result 0 reaches function exit without release
#
# MEASURED (2026-09-08, RelWithDebInfo), with the same class:
#
#   x = b.v; if isinstance(x, int): print(x + 1) ... correct (prints 8)
#   decode(b.v) through a narrowing function ....... correct
#   b.v == 7 / bool(b.v) / [b.v] into a list ....... correct
#   a method reading `self.v` into a local ......... correct
#   print(b.v) ..................................... the refusal above
#   str(b.v) ....................................... the same
#
# ⭐ WHY THE DIRECT SPELLING AND NOT THE LOCAL: the tag dispatch that renders
# a union releases the ACTIVE member's lane on each arm
# (`LyLong_DecRef(%5) {reference_release}`), and for a field read that lane is
# the SAME SSA value as the instance group's `%8#2` -- the retain the store
# put on the field. So the group has had one of its five values released
# already, and the placer will not put down the whole-group `__ly_dealloc_Box`
# at the normal exit; only the unwind cleanups call it. Through a local the
# rendered value is a copy the frame minted, and the group is untouched.
#
# ⛔ THIS IS THE SAME SHAPE as the owned half of "two tokens, one object": two
# ownership tokens live on ALIASED SSA values and the placer counts values,
# not objects. Three coordinated edits were tried there and each moved the
# complaint rather than closing it.
class Box:
    def __init__(self, v: "int | str") -> None:
        self.v: "int | str" = v


b = Box(7)
print(b.v)
