# differential: skip the OverflowError is the recorded refusal, not a wrong
# answer, and the runner would report it as a standing GAP every run
#
# REFUSED where CPython prints. A module-level `x: int = 1` is an unboxed i64
# cell, so module-scope arithmetic that grows it past 2**63 raises
# "OverflowError: int too large to convert to a native 64-bit integer" --
# while the identical loop over a LOCAL prints 1180591620717411303424.
#
# MEASURED:
#
#   x: int = 1 at module scope, x = x * 2 seventy times ... OverflowError
#   the same loop inside a function ......................  correct
#   x = 1 (unannotated) at module scope ..................  correct, because
#       an unannotated module int is not a cell at all -- its references
#       re-emit the literal (collectModuleGlobals), and the ⛔ note there
#       records that making it one would produce exactly this defect
#
# MECHANISM, located and measured: lowerGlobalGet/lowerGlobalSet
# (lowering/Passes/Runtime/Ops/GlobalOps.cpp) special-case builtins.int to an
# i64 cell and send every other contract down the object path, which holds
# the value group's words plus a retained reference -- what a boxed int needs.
# Sending int down it is one line, and it breaks the runtime's own
# stackguard_support.py: an object global's read retains and its write
# allocates, and that module has to be reachable from a signal handler.
#
# So the i64 cell is the async-signal-safe channel, not an optimization, and
# the repair needs a discriminator that does not exist yet -- something that
# separates a global a signal handler may touch from one only ordinary code
# touches. A per-module split is not enough: a user's own ctypes callback
# reading a boxed global is refused by verifyCallbackSignalSafety.
x: int = 1
for i in range(70):
    x = x * 2
print(x)
