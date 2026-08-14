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
# ⭐ IT IS THE `g.condition` HOLE, from the return side. A union's release is
# guarded by its tag, and the walks that place releases skip any group that
# carries a condition (Passes/Ownership.cpp). With one owning member the
# return ABI's owned-result lane covers it -- that is what
# `ly.ownership.owned_results` names, and why `int | None` works. With two
# there is no single lane to name, so the obligation stays conditional all the
# way to the exit and the verifier reports it rather than a placement doing
# something wrong.
#
# Which makes this the same open item as
# tests/probe/wb_union_carried_exit_release_leak.py, reached from the return
# instead of the loop: both need a release that is emitted under
# `cmpi eq(tag, activeTag)`, and neither is waiting on the guard itself.
#
# differential: skip refused; the point is the refusal


def pick3(n: int) -> int | str | None:
    if n == 0:
        return 5
    if n == 1:
        return "five"
    return None


print(pick3(0), pick3(1), pick3(2))
