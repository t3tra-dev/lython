# `try: ... return None ... finally:` with a declared `int | None` return is
# refused: "return value type through try/finally is not implemented yet".
# Every other return type through a finally works now (2026-09-01): a scalar
# constant, an empty container, or the unbound placeholder for a source class
# -- see cases/a_container_returned_through_a_finally.py.
#
# ⭐ WHY THE UNION IS THE ONE LEFT. The completion payload needs a value on the
# path that did NOT return, and `isSupportedFinallyReturnCarrierType` asks for
# a ContractType because that is what `emitDefaultReturnValue` and
# `defaultCompletionValue` can build. A union is lanes plus a TAG, so its
# default has to name an active member -- and the release on the discard path
# is tag-conditioned, which is the same guard
# `insertOwnedBlockArgumentReleases` skips for a union today
# ([[wb_union_loop_carried_borrow_overrelease]]).
#
# ⛔ NOT the unbound placeholder either: `py.unbound` builds one non-owning
# dead object, and a union payload needs one per member lane plus a tag that
# says which of them the release should look at.
def pick(flag: int) -> "int | None":
    try:
        if flag > 0:
            return flag
        return None
    finally:
        print("checked")


print(pick(1), pick(-1))
