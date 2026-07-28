# SILENT WRONG on 51fb04d. rc=0, no diagnostic, wrong number. CPython prints 7.
#
# A local written ONLY in a loop's `else` clause does not survive the loop's
# exit edge. The write is not dropped -- it is visible inside the `else` -- but
# the code after the loop reads a different incarnation of the name.
#
# Found by `tests/probe/tools/nestgrid.py`, which reports `while.else` as a
# region occurring in ONE file of 292 golden cases and `for.else` in two, and
# by `nestwitness.py`, which put a write in that region and a read after it and
# came back `WWWWW` on eleven cells. The cell LABELS (`while.else > try`,
# `while.else > listcomp`, ...) name the nested construct, and the nested
# construct turned out to be irrelevant: this file is the minimisation, and it
# nests nothing at all. That is worth stating plainly -- the grid pointed at
# an under-covered REGION and the cell name did not describe the mechanism.
#
# ============================================================
# MEASURED, one binary (RelWithDebInfo, 51fb04d), 3 reps each, load 12.4
# ============================================================
#   for/else, only the else writes `acc`, in a def ......... 0, want 7   <- here
#   for/else, only the else writes `acc`, at module level .. 0, want 7
#   while/else, only the else writes `acc` ................. 0, want 7
#   while/else, plain `acc = 7` instead of `acc += 7` ...... 0, want 7
#   for/else, the LOOP BODY ALSO writes `acc` .............. 10, want 10  OK
#   for/else, else only prints, writes nothing ............. correct
#   for/else with `break`, so the else is skipped .......... correct
#   write after the loop with NO else at all ............... 7, want 7   OK
#
# So the discriminator is narrow and exact: the name must be written in the
# `else` clause and NOT in the loop body. `break` handling, the else's own
# control flow, and side effects with no target name are all correct.
#
# WHY THE SUITE IS GREEN. `tests/golden/cases/loop_else.py` does exercise an
# else-clause write -- `total = total + 100` after `for v in [5, 6]: total =
# total + v` -- and prints 111. The loop body writes `total` too, which is the
# one spelling of the row above that works. Its `while` half writes `found` in
# the else, but that loop takes a `break`, so the else never runs. The
# corpus reaches this construct twice and misses the broken half both times.
#
# THE WRITE HAPPENS. This is the measurement that separates a lost store from a
# lost NAME, and it is why the header above does not say "the else clause is
# skipped" (which is what the first three rows on their own look like, and
# which I believed until I printed from inside the block):
#
#     def w():
#         acc = 0
#         for i in [1, 2]:
#             pass
#         else:
#             acc += 7
#             print("in else", acc)     # lyc: 7   -- correct
#         print("after", acc)           # lyc: 0   -- pre-loop value
#
# MECHANISM: INFERRED, NOT MEASURED. The shape matches the block-argument
# renaming already documented in `wb_forloop_handler_local_unwind.py` -- loop
# carried locals are threaded through the loop header as block arguments, and
# consumers that only rename forward leave a second incarnation reachable under
# the pre-loop name. An `else` block is the one region that is neither inside
# the loop nor past the point where the exit edge has been taken, so a write
# there landing on the pre-loop incarnation would produce exactly these eight
# rows. I did NOT dump the IR to confirm it, and `src/` is owned by other
# tracks this session, so no claim is made about which pass is responsible.
#
# NOT A GOLDEN, because there is no repair to lock in and a red golden is not
# something to commit. When it is fixed, the eight rows above are the test.
def w() -> int:
    acc = 0
    for i in [1, 2]:
        pass
    else:
        acc += 7
    return acc


print(w())
