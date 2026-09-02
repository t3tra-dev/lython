# A class field whose type is a union with TWO owning members cannot be
# compiled. `int | None` works and always has; `int | str` does not, and the
# difference is that every union this compiler had reached had a member that
# carries no lanes.
#
# MEASURED (2026-09-02, RelWithDebInfo, today's tree):
#
#   self.v: "int | None" = 1 ..................... correct (prints 1)
#   self.v: "int | str"  = 1 ..................... ly.ownership.owned_local_object
#                                                  marks a value this frame never
#                                                  acquired
#   the same at function scope ................... same
#   read in a method instead of at module scope .. same
#   c.v = "a" before the read .................... same
#   def f(v: "int | str") (a PARAMETER) .......... correct (prints 1 / a)
#   returning "int | str" from a function ........ correct
#
# ⭐ WHY THE FIELD AND NOT THE PARAMETER: the instance is built before
# `__init__` runs, so the field starts at a DEAD value, and
# `materializeDeadObjectValueImpl` (Runtime/ABI/RuntimeABI.cpp) builds one by
# zeroing every lane and pointing the tag at a member that owns nothing --
# `None` in every Optional. `int | str` has no such member, so the tag stayed
# at member 0, the frame claimed a heap placeholder it had zeroed rather than
# initialised, and the ownership verifier refused the token. A parameter and a
# return never pass through a dead value.
#
# ⛔ MAKING THE DEAD LANES STATIC AND IMMORTAL IS NOT ENOUGH, measured: the
# refusal moves to "owned resource from builtin.unrealized_conversion_cast
# result 0 reaches function exit without release" on the INSTANCE. With a
# two-owning-member union field the class's ABI bundle carries the union lanes
# beside the header (`__ly_dealloc_C` takes 5 arguments where the `int | None`
# spelling takes 1), and the release placer puts down no release for that
# group at the normal exit -- only the unwind cleanups call the deallocator.
# So the repair is at least two: a dead value for a union with no empty
# member, and a release for a class group whose values include union lanes
# that are separately owned elsewhere in the frame.
#
# ⛔ AND THE TAG CANNOT SIMPLY NAME NO MEMBER: the aggregate release of the
# field's previous contents is emitted per member and unconditionally
# (`LyLong_DecRef` and `LyUnicode_DecRef` both, in the same block), so an
# out-of-range tag would leak both lanes rather than release neither. Making
# that release tag-conditioned is the same missing machinery
# wb_union_loop_carried_borrow_overrelease and wb_return_a_union_through_finally
# both end at.
class Box:
    def __init__(self) -> None:
        self.v: "int | str" = 1


b = Box()
print(b.v)
b.v = "later"
print(b.v)
