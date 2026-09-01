# A class whose ONLY field is written `= None` inside a loop leaves the
# constructed object unreleased:
#
#     owned resource from builtin.unrealized_conversion_cast result 0 reaches
#     function exit without release, transfer, or owned return
#
# reported in __main__, at `c = C()` -- reading the field is not needed.
# Everything one step away compiles (field-region sweep, 75 programs):
#
#     self.v = None   at __init__'s top level ......... ok
#     self.v = None   in BOTH arms of an if/else ..... ok
#     self.v = 1      in the same loop ............... ok
#     self.v = [1, 2] in the same loop ............... ok  (13 other kinds too)
#
# so it is the pair (NoneType field, written only inside a region with a back
# edge or a handler): the `for`, `while` and `try` spellings all fail and the
# if/else one does not.
#
# ⭐ The class op is IDENTICAL between the working and failing spellings --
# `field_types = [!py.literal<None>]` in both -- so nothing about the FIELD
# decides it. What differs is `__init__`'s block structure, and the release
# that goes missing is the CALLER's, on the object `__init__` returned. A
# NoneType field has no runtime lanes of its own, so the object is all header:
# the shape where the ownership walk has the least to hold on to.
#
# ⛔ Reachable only since fields assigned inside a region became fields at all
# (2026-09-02); before that this program was "class C has no field 'v'".
class C:
    def __init__(self) -> None:
        for _ in range(1):
            self.v = None


c = C()
print(c.v)
