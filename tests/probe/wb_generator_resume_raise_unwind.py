# TWO OPEN DEFECTS on df48b61, both isolated, NEITHER FIXED. They share one
# shape: an exception raised INSIDE a generator body, escaping the resume.
#
# ⚠️⚠️ THE AXIS BELOW IS WRONG ON bcfbbf9 -- see
# `wb_forloop_handler_local_unwind.py`, which has NO generator and raises NO
# exception and still SIGSEGVs 5/5. Two rows of the minimisation table have
# moved since it was taken:
#
#     same try/except/accumulator, NO generator ...... recorded OK, now CRASH
#     handler returns a constant instead of reading .. recorded OK, now REFUSED
#
# The real trigger is `for` loop (over anything) + `try`/`except` in the same
# function + a local written in the loop and read in the handler. The generator
# is not part of it. The root cause -- ONE cell object tracked as TWO ownership
# entities across a loop-header block-argument rename, with all three consumers
# handling the rename forward-only -- is written up in that other file, together
# with the measured ablation of a four-part repair that fixes every spelling and
# refuses 80 of 490 tests, and is therefore NOT SHIPPED.
#
# Keep this file: defect (2) below (the bounded leak) is recorded nowhere else,
# and its numbers were taken on a frozen compiler that no longer exists.
#
# This file is the CRASHING spelling. Run it: it SIGSEGVs. CPython prints 100.
#
# ============================================================
# (1) SIGSEGV when the escaping exception is CAUGHT  -- pre-existing
# ============================================================
# Deterministic (3/3), and present on 4699488 BEFORE the generator-argument
# retain fix (e39fe8a), so that fix did not introduce it.
#
#   EXC_BAD_ACCESS address=0x0 in Ly_IncRef, called from f.
#
# From the refcount-elision IR: `for v in gen(...)` puts the loop-carried local
# into a cell (`memref<16xi64>`). The unwind pads for the resume call
# (`__ly_unwind_cleanup_4/_5`) end with `__ly_dealloc___ly_cell_1(...)`, which
# runs `LyObject_ReleaseStorageToZero` and `memref.dealloc`s the cell -- and
# then branch to the `except` block, WHICH LOADS THAT CELL and retains the
# loaded header. The slot reads back 0, so `Ly_IncRef` dereferences null. The
# post-handler return block reads the same freed cell and deallocates it again.
#
# So the pad destroys the enclosing try's locals before the handler that is
# going to read them runs. The pad is built as if the exception were leaving
# the function; the handler is in the same function.
#
# MINIMISATION (8 spellings, one frozen compiler):
#   handler reads the accumulator ........................ CRASH
#   handler reads nothing, returns a constant ............ OK
#   handler ASSIGNS the accumulator instead of reading ... REJECT (lowering)
#   `next()` in try, no `for` loop ....................... OK
#   generator takes ints, no object argument ............. CRASH
#   explicit `raise` before the first yield .............. REJECT (lowering)
#   same try/except/accumulator, NO generator ............ OK
#   generator, exception raised in the LOOP BODY ......... CRASH
# So: needs `for` over a state-machine generator + a local both written in the
# loop and read in the handler. Not about object arguments (int args crash too).
#
# ⚠️ A crash makes `leaks --atExit` emit NOTHING. An unguarded parser reads that
# as "0 leaks". Any harness reporting 0 for the caught case should be re-checked
# against the process exit status before the 0 is believed.
#
# ============================================================
# (2) BOUNDED leak when the escaping exception is UNCAUGHT
# ============================================================
# `leaks --atExit`, baseline `print(0)` = 1 root / 540672 B subtracted:
#
#   generator arg = list[10, 0], raises in body ....  6 roots / 11152 B
#   generator arg = ints, raises in body ..........   3 roots /   768 B
#   generator arg = list of 16, raises in body ....  18 roots / 11920 B
#   raises on the SECOND resume ...................   7 roots / 11216 B
#   TWO generators, second raises .................   7 roots / 11216 B  (flat)
#   generator runs clean, no exception ............   0 roots /     0 B
#   uncaught exception, NO generator ..............   3 roots /   192 B
#
# Flat across 1st-vs-2nd resume and 1-vs-2 generators, so bounded, not per
# iteration. The 768 B int-argument figure is exactly 640 (generator storage)
# + 64 (exception) + 64 (message): THE GENERATOR STORAGE ITSELF LEAKS, and it
# takes its frame contents with it -- the argument list, that list's payload
# (a fixed 10240 B buffer: 2 elements and 16 elements both give 10240) and the
# boxed elements. Root count scales with what the frame holds, not with time.
#
# THE OBVIOUS EXPLANATION IS WRONG, MEASURED TWO WAYS:
#
#   "the drop finalizer is not called on the escape path" -- FALSE. A generator
#   that SUSPENDS and is then abandoned by an uncaught exception from the loop
#   body runs its `finally` (prints, matching CPython) and leaks only
#   2 roots / 128 B, the same residue as any uncaught exception. The finalizer
#   and the drop hook work. This test is inlining-proof; a symbol breakpoint is
#   not, and must not be used to claim a release "never ran".
#
#   The IR also already contains the release: the resume call's unwind pad is
#   `__ly_unwind_cleanup_5(list, generator, ...)`, which calls `LyList_DecRef`
#   AND `LyGenerator_DecRef`. Running it would drive the generator to zero, run
#   the finalizer, and free the list.
#
# What the numbers say instead: the list is released by that same pad, and the
# list LEAKS -- so the pad guarding the RESUME CALL is not entered when the
# resume itself raises, while the pad guarding the LOOP BODY is (case above,
# everything freed). That is the difference to chase, and it is the same pad
# family that defect (1) mis-orders.
from typing import Iterator


def gen(xs: list[int]) -> Iterator[int]:
    yield xs[0] // xs[1]
    yield xs[0]


def f() -> int:
    xs = [10, 0]
    total = 0
    try:
        for v in gen(xs):
            total += v
    except ZeroDivisionError:
        total += 100
    return total


print(f())
