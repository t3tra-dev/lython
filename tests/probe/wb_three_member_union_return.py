# OPEN. A union with TWO owning members, returned:
#
#     conditionally owned resource from @pick3 result 4 reaches function exit
#     without tag-conditioned release, transfer, or owned return
#
# BISECTED against ./build/bin/lyc:
#
#   int | str | None, returned and printed ..... refused   <- this file
#   int | None, the same shape ................. OK (cases/union_renders_by_tag)
#   str | None, the same shape ................. OK
#   int | str | None built and NARROWED in the
#     same frame, never returned ............... OK
#
# So it is the two OWNING members crossing the return boundary, not the third
# member and not the rendering: the renderer handles a three-member chain, and
# every two-member spelling of the same program runs.
#
# ⭐ LOCALIZED 2026-08-15 to the exact declaration, which is one field:
#
#     struct ReturnedStaticObjectSummary {     // Runtime/Model/Bundles.h:372
#       mlir::Type objectContract;             //  <- ONE
#       unsigned resultIndex = 0;
#     };
#
# `buildReturnedStaticObjectSummaries` (Evidence/Returned.cpp) walks the
# returns collecting the object contract each one produces, and the moment a
# SECOND distinct one appears it sets `allReturnsSummarized = false` and emits
# no summary at all. No summary means no owned-result lane, which is why the
# obligation stays conditional all the way to the exit -- the verifier is
# reporting the absence correctly.
#
# ⭐ WHAT THE LANE ACTUALLY IS, measured from the two ABIs, because it is not
# the union's own layout and that is the surprise:
#
#   str | None      -> (i64 tag, str hdr, str bytes, str hdr, str bytes)
#                      owned_results = [3], contracts = ["builtins.str"]
#   int | str | None-> (i64 tag, int hdr, int meta, int digits,
#                       str hdr, str bytes)
#                      no owned_results at all
#
# The two-member ABI carries the str lanes TWICE: once as the union's own
# member layout (borrowed) and once as an appended static-object evidence lane
# (owned). So the repair is not "mark the union's member lanes owned" -- it is
# "append one evidence lane per owning member", and the attribute side already
# takes a list (`owned_results` is a DenseI64Array and
# `owned_result_contracts` an ArrayAttr; the generator clone path in
# CallableABI.cpp already writes several).
#
# ⛔ WHY IT IS STILL NOT A SMALL CHANGE, and the four sites are the estimate:
#
#   1. Evidence/Returned.cpp -- collect a SET of contracts and order it by the
#      union's member order (not by the order the returns were walked, or the
#      ABI is not a function of the type).
#   2. ABI/CallableABI.cpp:1082 -- loop the append instead of doing it once.
#   3. ABI/Returns.cpp:452 -- per member, fill the lane from the active member
#      when it matches and from a dead placeholder otherwise. Already written
#      for one; the loop is mechanical.
#   4. Ops/FunctionTargetCalls.cpp:679 -- THE HARD ONE. `RuntimeBundle` has a
#      single `boxedObject` slot, and with two owning members which one holds
#      the token is a RUNTIME question the tag answers. The caller needs a
#      per-lane conditional bundle and a release emitted under
#      `cmpi eq(tag, activeTag)` -- which is the same missing mechanism as
#      tests/probe/wb_union_carried_exit_release_leak.py, reached from the
#      return instead of the loop. Neither is waiting on the guard itself;
#      both are waiting on a bundle that can be conditionally owned.
#
# differential: skip refused; the point is the refusal


def pick3(n: int) -> int | str | None:
    if n == 0:
        return 5
    if n == 1:
        return "five"
    return None


print(pick3(0), pick3(1), pick3(2))
