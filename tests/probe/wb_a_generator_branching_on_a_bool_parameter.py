# A generator that branches on a BOOL PARAMETER around a yield is refused:
#
#     source generator next lowering currently supports only straight-line
#     pure int yield bodies
#
# -- which is true of that tier and is not why the program came down to it.
# The boundary is sharp, and every neighbour compiles:
#
#     def g(flag: bool): yield 1 ............................. ok
#     def g(flag: bool): yield 1; yield 2 .................... ok
#     def g(n: int):  if n > 0: yield 1 else: yield 2 ........ ok
#     def g(n: int):  flag = n > 0; if flag: yield 1 else: ... ok
#     def g(flag: bool): if flag: yield 1 else: yield 2 ...... REFUSED
#     def g(flag: bool): if flag: yield 1; yield 2 .......... REFUSED
#     def g(n, flag: bool): for i in ...: if flag: yield i ... REFUSED
#
# so it is neither the branch nor the bool: it is a bool that arrives as a
# PARAMETER and decides a branch around a yield. A bool LOCAL derived from a
# comparison is fine.
#
# ⛔ MEASURED AND WRONG: extending the state machine's rematerialization (which
# already copies operand-free pure ops to their uses, for the `arith.constant
# true` a `continue` leaves live) to ops whose operands are all clone ENTRY
# ARGUMENTS. `py.bool(%arg0)` is exactly that shape, and rematerializing it
# changed nothing.
#
# ⭐ THE CAUSE IS THE PARAMETER'S SHAPE, not the branch. `builtins.bool`'s
# manifest value shape is a BARE i1:
#
#     func.func private @LyBool_Shape() -> i1
#         attributes {ly.runtime.contract = "builtins.bool", ly.runtime.shape}
#
# and `generatorLaneParts` requires every part to be a rank-1 MEMREF, because
# a frame slot holds (pointer, size) word pairs. So a bool parameter has no
# frame lane, `argumentsEligible` goes false in
# `buildGeneratorResumeCloneSignatures`, and the state machine skips the
# generator entirely -- WITHOUT recording a decline reason, which is why the
# tier below has nothing to append to its own message. A str, float or list
# parameter deciding the same branch compiles, and all three have memref
# shapes.
#
# THE SHAPE OF THE REPAIR: a bool lane stored as one i64 word (0/1), widened
# on store and truncated on load. The frame already has a non-memref lane kind
# -- `lane.isInt`, an i64 plus an i1 valid flag -- so the mechanism exists; it
# is consulted at 14 sites, which is the cost. Additive: a bool parameter has
# no lane at all today, so nothing that works now would change.
def g(flag: bool):
    if flag:
        yield 1
    else:
        yield 2


print(list(g(True)), list(g(False)))
