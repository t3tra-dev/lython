# OPEN: a union-typed field STORED TWICE, where one of the stores takes a
# parameter, is refused by the ownership walk:
#
#   def f(n: int) -> bool:
#       b = Box(1)          # Box.v: "int | str"
#       b.v = n
#       b.v = n + 1
#   # borrowed entry argument 0 of @f is released or transferred without a
#   # prior retain
#
# MEASURED 2026-09-08 (RelWithDebInfo), `Box.v: "int | str"` throughout:
#
#   b.v = n (a parameter), once ..................... correct
#   b.v = "a"; b.v = 2 (constants, module scope) .... correct
#   b.v = "a"; b.v = "b" inside a function .......... correct
#   b.v = n; b.v = n + 1 (a parameter, then again) .. the refusal above
#   b = Box(text); b.v = 1; b.v = text (a str param)  the same refusal
#   text = "z"*4096 local; the same three stores .... "released owned resource
#                                                      from @LyUnicode_Mul is
#                                                      used after release"
#   the same shapes with an `int | None` field ...... correct
#
# ⭐ WHAT THE THREE HAVE IN COMMON is the SECOND store, which releases what the
# field held. A union field keeps its members INLINE in the instance's SSA
# lanes, so "what the field held" is whatever SSA value the previous store
# spliced in -- the caller's borrowed argument, or the very value being stored
# again -- and releasing it is releasing something this frame does not own.
# A boxed field has no such alias: the box holds a payload, and the release
# reads the box.
#
# ⭐ AND THE SECOND STORE LEAKS WHERE IT IS ACCEPTED. `leak_sweep.py` over
#
#     b = Box("abcdefghij")     # Box.v: "int | str"
#     b.v = 1
#
# reports 1 alloc / 58 B -- the whole str. The IR has the field's retain
# (`Ly_IncRef ... aggregate_retain = "builtins.str:class.v"`) and the field's
# release at the overwrite (`LyUnicode_DecRef ... aggregate_release`), and
# NOTHING releases the constructor's own temporary: the two tokens live on ONE
# SSA value, so the planner read the aggregate release as discharging both.
# The same program with a `str | None` field, and with a plain `str` field, is
# clean -- both are boxed, so the field's reference is a word in the box and
# not an alias of the frame's value.
#
# ⛔ SO THIS IS THE SAME MISSING MECHANISM as
# wb_a_union_field_written_inside_a_region and the "expands to 5 physical
# values" refusal that keeps such an object out of a list. One storage
# decision, four symptoms.
class Box:
    def __init__(self, v: "int | str") -> None:
        self.v: "int | str" = v


def f(n: int) -> bool:
    b = Box(1)
    b.v = n
    b.v = n + 1
    return isinstance(b.v, int)


print(f(3))
