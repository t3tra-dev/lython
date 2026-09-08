# OPEN. An unannotated function that accumulates its result into an EMPTY
# container literal answers `list[builtins.object]` in the fixpoint round
# before its own parameter is known, and the round that learns the parameter
# then collides with what the first one left behind -- but only once another
# function's PARAMETER consumes the result:
#
#     def double(xs):
#         out = []
#         for x in xs:
#             out.append(x * 2)
#         return out
#     def total(xs: "list[int]") -> int: ...
#     print(total(double([1, 2])))
#     # cannot unify !py.contract<"builtins.int">
#     #        with !py.contract<"builtins.object">
#
# MEASURED 2026-09-09, RelWithDebInfo, on the tree that repaired the module
# scope (cases/a_module_name_passed_to_an_inferred_parameter). Two rows moved
# there; the rest read the same before and after, so this is its own mechanism.
#
#   `print(double([1, 2]))` on its own ......................... correct
#   inferred -> inferred, SCALAR result ........................ CORRECT NOW
#   annotated -> inferred, container result .................... CORRECT NOW
#   inferred -> annotated, container result .................... refused
#   inferred -> inferred, container result ..................... refused
#   the same chain through a module name ....................... refused
#   the producer's result written `return xs + xs` ............. correct
#   the producer's result annotated `-> "list[int]"` ........... correct
#   the producer's ACCUMULATOR annotated `out: "list[int]"` .... correct
#   a str or an int result, accumulated the same way ........... correct
#
# ⭐ TWO THINGS TOGETHER, NEITHER ALONE. The empty literal alone compiles --
# `print(double([1, 2]))` prints `[2, 4]` -- and a chain alone compiles, as the
# scalar and expression-result rows show. What does not survive is a walk that
# had to answer for `out = []` while `xs` was still a variable AND a consumer
# that pins the answer before the next round can revise it.
#
# ⛔ The guard that fixed the same shape for the RESULT type does not reach
# this: `functionSignature` now declines to bind a walked result while the
# function's own parameters are unresolved (TypeSystem.cpp, "AND NOT WHILE
# THIS FUNCTION'S OWN PARAMETERS ARE STILL UNKNOWN"), and the element type is
# bound by the container rule inside the body walk, which never passes through
# that test.
#
# ⛔ Not "make the empty literal answer a variable": that variable would join
# with every later append in the same walk and reach the frame lane of any
# generator holding the container. The shape of a repair is to keep the
# container's element OPEN for as long as the parameters are -- which is the
# same question the empty-container rule already answers for a local, asked one
# scope out. Scoped, not built.
#
# ⛔ And the message is the compiler's, not the program's: "cannot unify
# !py.contract<...>" names the inference store. Whatever fixes this owes the
# reader a sentence about the two functions instead.
def double(xs):
    out = []
    for x in xs:
        out.append(x * 2)
    return out


def total(xs):
    n = 0
    for x in xs:
        n += x
    return n


print(total(double([1, 2])))
