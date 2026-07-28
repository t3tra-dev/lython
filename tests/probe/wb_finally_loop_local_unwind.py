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
# ⚠️ NO REPAIR, so no golden. `wb_forloop_handler_local_unwind.py` records a
# four-part attempt that turns the `except` form into a correct answer at the
# cost of 80 of 490 tests REFUSED, and its root cause -- one object tracked as
# two ownership entities across the loop's back edge -- is in `src/`, which
# other tracks own this session. Nothing in `src/` was touched to produce this
# file.
def w() -> int:
    acc = 0
    try:
        acc += 1
    finally:
        for j in [1, 2]:
            acc += j
    return acc


print(w())
