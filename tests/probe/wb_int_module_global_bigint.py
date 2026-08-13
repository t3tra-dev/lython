# FIXED 2026-08-13. Kept as the reproducer, and as the record of what the
# repair had to decide before it could exist.
#
# WAS: refused where CPython prints. A module-level `x: int = 1` was an
# unboxed i64 cell, so module-scope arithmetic that grew it past 2**63 raised
# "OverflowError: int too large to convert to a native 64-bit integer" --
# while the identical loop over a LOCAL printed 1180591620717411303424.
#
# MEASURED, at the time:
#
#   x: int = 1 at module scope, x = x * 2 seventy times ... OverflowError
#   the same loop inside a function ......................  correct
#   x = 1 (unannotated) at module scope ..................  correct, because
#       an unannotated module int is not a cell at all -- its references
#       re-emit the literal (collectModuleGlobals), and the ⛔ note there
#       records that making it one would produce exactly this defect
#
# WHY IT TOOK A DECISION AND NOT A PATCH: one unboxed i64 cell was serving two
# different things. A Python integer, which must grow; and a machine address,
# which must stay a word because a ctypes callback reads it and may not
# allocate. Boxing every int global is one line, and it left
# `examples/ctypes_signal.py` correctly refused by verifyCallbackSignalSafety
# -- the policy working, on a program with nowhere else to put its address.
#
# THE REPAIR, in the order it had to happen:
#   1. an address spelling: a `ctypes.c_void_p` module global is the machine
#      word (lowerAddressGlobalGet/Set), and `ctypes.cast(p, PROTO)` reaches
#      the code at it -- CPython's own spelling, since `PROTO(p)` takes an
#      integer and rejects a c_void_p;
#   2. `int` globals box, keyed on a `ly.global.boxed` mark the emitter puts
#      on the two ops that reach the MODULE-GLOBAL population (default cells
#      and class-attribute slots ride the same ops and must keep the word);
#   3. `EmitOptions::runtimeInternal` exempts runtime/lib/*.py, whose globals
#      hold libc pointers read from the stack-guard's signal handler.
#
# tests/golden/cases/module_global_int_grows.py is the golden.
#
# differential: skip kept as a reproducer; the golden is the live assertion
x: int = 1
for i in range(70):
    x = x * 2
print(x)
