# FIXED 2026-08-15 by the repair in `wb_forloop_handler_local_unwind.py`, and
# this file is why that repair can be believed. Everything the table below
# established as WIDER than the `except` form -- the `finally` clause, `while`,
# module scope, a local nobody reads again -- is one program point: the borrow
# edge into a loop-carried merge, refused a retain because the spellability
# predicate asked about the `memref.alloc` instead of about where the retain
# goes. Nothing in the repair mentions `except`, `for`, a frame or a read, so
# the seven crashing rows and the five clean ones both fall out of it.
#
# ⭐ AND THE PREDICTION HELD BOTH WAYS. The five `.....` rows are programs that
# never reach the merge, so they were already correct and had to stay correct;
# 710/710 in both builds says they did.
#
# golden: tests/golden/cases/loop_in_try_cell_merge.py (red-checked), whose
# `for_finally` is this shape with the loop in the try body and `while_rebind`
# is the `while` row.
#
# ============================================================
# SHIPPED SIGSEGV on 51fb04d. rc=139, deterministic 5/5. CPython prints 4.
# NO `except` CLAUSE ANYWHERE, and nothing outside the `try` statement reads
# the local the loop writes.
#
# This file exists because the trigger recorded in
# `wb_forloop_handler_local_unwind.py` is NARROWER than the defect. That file's
# table says the crash needs all three of
#
#     (a) a `for` loop, over anything, and
#     (b) a `try`/`except` in the same function, and
#     (c) a local written in the loop body and READ IN THE HANDLER
#
# and names (c) as the condition that selects between crashing and being
# refused. Replace the `except` with a `finally` and (c) stops being required:
# a `try`/`finally` whose loop-written local nobody reads afterwards still
# crashes. So the `finally` form is strictly WIDER than the `except` form, and
# a repair validated only against `except` would leave this shipping.
#
# Found by `tests/probe/tools/nestgrid.py`, which named `try.finally > for` as
# an empty cell in all four corpora (`try.finally` occurs in 16 of 292 golden
# cases, `for` in 87, and no file puts one inside the other), and by
# `nestwitness.py`, which synthesised the program below and got `XXXXX`.
#
# ============================================================
# MEASURED, one binary (RelWithDebInfo, 51fb04d), 5 reps each.
# Load averages 41.8 -> 78.5 across the two batches: high, and irrelevant to
# these rows. A contended machine can fabricate a timeout but not a SIGSEGV
# and not a matching answer (rfc/stdlib-semantics.md 13c). Zero timeouts were
# observed; every row below is 5/5 identical.
# ============================================================
#   loop in the FINALLY, try body writes acc, acc read after ..... XXXXX  <- here
#   loop in the FINALLY, try body does not touch acc ............. XXXXX
#   loop in the FINALLY, NOTHING reads acc afterwards (`return 99`) XXXXX
#   loop in the TRY BODY, with a finally and no except ........... XXXXX
#   loop in the TRY BODY, with an except that reads acc .......... XXXXX
#   same, but a `while` instead of a `for` ...................... XXXXX
#   loop in the FINALLY at MODULE level, no function at all ...... XXXXX
#
#   try/finally with NO loop inside it ........................... .....
#   loop in the FINALLY that writes no local (`print(j)` only) .... .....
#   loop BEFORE the try/finally, outside it ...................... .....
#   loop AFTER the try/finally, outside it ....................... .....
#   loop before a try/EXCEPT, outside it ......................... .....
#
# So the `finally` trigger is:
#
#     (a) a loop -- `for` or `while` -- lexically INSIDE a `try` statement,
#         in either its body or its finally clause, and
#     (b) that `try` has a `finally`, and
#     (c') a local written in the loop body.
#
# (c') is where it differs: the `except` form needs that local to be read again
# in the handler, and this one does not need it to be read at all. Two other
# conditions the older file left open are closed here: the loop keyword does
# not matter, and a function frame is not required -- module level crashes too.
#
# THE FRAME IS NOT THE DISCRIMINATOR, which is worth recording because it was
# my first guess and it was wrong. `tests/golden/cases/dict_iteration_views.py`
# does put a `for` in a `try` body and is green, and the difference is NOT that
# it is at module level -- the module-level rows above crash. It is green
# because its handler prints the caught exception rather than the loop's
# accumulator, i.e. it lacks (c). nestgrid.py counts that edge for exactly this
# reason.
#
# ⛔ THE PARAGRAPH THAT STOOD HERE SAID "NO REPAIR, so no golden", and pointed
# at the four-part attempt that costs 80 of 490 tests. It is kept as history in
# the other file and it is not what shipped: two ownership entities over one
# allocation is the ordinary shape of a merge, and the repair pays the second
# one rather than deleting it.
def w() -> int:
    acc = 0
    try:
        acc += 1
    finally:
        for j in [1, 2]:
            acc += j
    return acc


print(w())
