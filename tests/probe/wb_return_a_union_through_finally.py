# FIXED (2026-09-02) for the value returns; what is left is the BORROWED
# PARAMETER. `def pick(flag: int) -> "int | None"` returning 7 or None through
# a finally compiles (golden: an_optional_returned_through_a_finally); the
# program below returns the PARAMETER, and that is:
#
#     borrowed entry argument 0 of @pick is returned as owned without a
#     dominating retain
#
# ⭐ THE UNION'S DEFAULT IS ITS None MEMBER. The completion payload needs a
# value on the path that did NOT return, and the payload is lanes plus a TAG,
# so the default has to name an ACTIVE member -- and the release on the discard
# path is tag-conditioned. None is the member that makes both free: it is a
# non-owning singleton, so the tag-conditioned release does nothing whichever
# way it is read. Both twins (`emitDefaultReturnValue` in the emitter,
# `pythonDefault` in Ops/TryOps.cpp) pick it, and only for a union that HAS
# one.
#
# ⛔ WHAT REMAINS: `insertBorrowedReturnRetains` asks whether the returned
# value is derived from an entry argument, and for a union through a finally
# the returned value is a MERGE of the real return and the synthesized default
# -- the default is not derived from anything, so the walk answers no and the
# retain never goes in. The verifier walks PATHS, so it sees the one path that
# does return the borrow. The retain belongs where the borrow is yielded INTO
# the completion payload, not at the func.return, which is a different question
# from the one that walk asks today.
#
# ⛔ A union of two OWNING members has no default at all: the discard-path
# release would have to be tag-conditioned for real, which is the shape
# recorded in wb_union_loop_carried_borrow_overrelease.
def pick(flag: int) -> "int | None":
    try:
        return flag
    finally:
        print("checked")


print(pick(1))
