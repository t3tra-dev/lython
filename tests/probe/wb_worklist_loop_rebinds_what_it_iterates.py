# FIXED 2026-08-30. The worklist loop -- iterate a carried list, build the next
# one, rebind -- was refused: "owned resource from @LyList_FromLength result 0
# reaches function exit without release, transfer, or owned return".
#
# ⭐ THE WALK PLACED NOTHING AT ALL, which is why the exit edge looked like the
# missing piece. `releaseOwnedGroupByLiveness` pins liveness for a group's
# VIEWS as well as its values, and for a view used inside a REGION op it
# answered "no ancestor in this region" whenever the group also had a consuming
# call -- and the caller reads that as "place no releases". The element load of
# `for task in pending` is exactly that: a `memref.load` inside an `scf.if`.
# So the back edge's decref-on-replace (which another walk emits) was the only
# release in the function, and the last list reached the exit owned.
#
# The repair is one condition: a view read inside a region op pins at the
# region op, consuming call or not. The null bail is kept for a use with no
# ancestor in this region at all, which is what the check was written for.
#
# ⭐ HOW IT WAS FOUND, after the earlier note guessed wrong twice: print the
# candidate set and then the RELEASE SITES the walk chose. `edges=0` with
# `after=0 beforeTerm=0` for a group that plainly dies says the walk returned
# early, and a print at each early return named the line in one run. The two
# earlier theories (the view-forwarding drop, the exit edge) were both about
# code that never ran for this group.
#
# ⛔ The condition it removed had no comment and dates to a bulk commit, so
# there was no rationale to weigh -- one more of the shape recorded in
# [[lython-stale-rationale]].
#
# Golden: cases/a_worklist_loop_rebinds_what_it_iterates.
def run() -> None:
    pending = ["a"]
    for n in range(2):
        for task in pending:
            pass
        pending = []


run()
