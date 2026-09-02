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
# changed nothing -- so the i1 is not what the state machine declines on, and
# the decline happens before or after the lane scan rather than in it.
#
# ⭐ The message cannot say more yet either: `generatorDeclineReasons` is only
# populated by the no-lane-contract arm, so the straight-line tier has nothing
# to append. Whatever declines this shape does so silently, and finding it is
# the first half of the repair.
def g(flag: bool):
    if flag:
        yield 1
    else:
        yield 2


print(list(g(True)), list(g(False)))
