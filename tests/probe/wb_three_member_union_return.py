# FIXED 2026-08-15. A union with TWO owning members, returned, was refused:
#
#     conditionally owned resource from @pick3 result 4 reaches function exit
#     without tag-conditioned release, transfer, or owned return
#
# BISECTED before the repair:
#
#   int | str | None, returned and printed ..... refused   <- this file
#   int | None, the same shape ................. OK (cases/union_renders_by_tag)
#   str | None, the same shape ................. OK
#   int | str | None built and NARROWED in the
#     same frame, never returned ............... OK
#
# ⭐ THE CAUSE WAS ONE FIELD, and the two-member ABI is what pointed at it:
#
#     struct ReturnedStaticObjectSummary {     // Runtime/Model/Bundles.h
#       mlir::Type objectContract;             //  <- ONE
#       unsigned resultIndex = 0;
#     };
#
# `buildReturnedStaticObjectSummaries` collects the object contract each return
# produces and abandons the whole summary the moment a SECOND distinct one
# appears. No summary meant no owned-result lane, so the obligation stayed
# conditional all the way to the exit and the verifier reported the absence
# correctly.
#
# ⭐ AND THE ABIs SAID WHAT THE ONE-MEMBER PATH ACTUALLY DOES, which is what
# kept the repair from being the obvious one:
#
#   str | None      -> (i64 tag, str hdr, str bytes, str hdr, str bytes)
#                      owned_results = [3], contracts = ["builtins.str"]
#   int | str | None-> (i64 tag, int hdr, int meta, int digits,
#                       str hdr, str bytes)
#                      no owned_results at all
#
# The two-member ABI carries the str lanes TWICE -- once as the union's own
# member layout, borrowed, and once as an appended static-object evidence lane,
# owned. So the summary is not a description of the union's layout; it is an
# extra copy bolted beside it.
#
# ⛔ Why NOT extend that summary to a LIST of contracts, which is what the
# attribute already takes (`owned_results` is a DenseI64Array and the generator
# clone path writes several): it appends a duplicate lane per member, and the
# caller would then need one conditionally owned bundle per duplicate while
# `RuntimeBundle` has a single `boxedObject` slot. That is the wall the earlier
# note called "the missing mechanism".
#
# ⭐ THE REPAIR GOES ROUND IT. The union's OWN layout already puts each member
# after the tag, and `collectTypedResourceGroups` (common/Ownership.cpp)
# already walks those lanes and already stamps each group with its
# `OwnershipCondition{tag, memberIndex, memberCount}`. Every piece of the
# conditional machinery was present; nothing named the offsets. So a union
# result with more than one owning member now declares its own member lanes as
# owned results (`prepareCallableFunctionABIs`), and no second copy exists to
# need a second bundle.
#
# "Owning" is asked of the manifest, not guessed from the type: a member counts
# only when some `ly.runtime.deallocator` claims its contract, so a bool or a
# None member names no lane.
#
# ⛔ WHAT IS STILL THE ONE-MEMBER PATH: a union with exactly one owning member
# keeps the static-object evidence lane it has always had. Two mechanisms for
# one job is a smell, and unifying them is a real simplification -- but the
# evidence lane also carries the PROTOCOL and coroutine returns through the
# same field, so removing it is not a union change. Measured, not merged.
#
# golden: tests/golden/cases/union_return_two_owning_members.py (red-checked;
# also in LYTHON_LEAK_GATE_CASES, since which lane gets released is invisible
# to stdout)


def pick3(n: int) -> int | str | None:
    if n == 0:
        return 5
    if n == 1:
        return "five"
    return None


print(pick3(0), pick3(1), pick3(2))
