# The type-dispatch renderer every serializer is written as. `isinstance(v, T)`
# narrows an `object`-typed value for every builtin T except `int` and `bool`.
#
# MEASURED (2026-09-02, RelWithDebInfo, today's tree), `def f(value: object)`
# with `if isinstance(value, T):` and a use of the narrowed value:
#
#   str, float, list, dict, tuple, bytes, set ........ correct
#   int (`value + 1`) ................................ static type
#                                                      `builtins.object` does
#                                                      not provide '__add__'
#   bool (`"y" if value else "n"`) ................... builtins.object.__bool__
#                                                      is declared by the
#                                                      standard-library contract
#
# ⭐ THE TEST IS RIGHT; ONLY THE VIEW IS WITHHELD, and the reason is measured in
# `analyzeIsInstance` (EmitterSupport.cpp): narrowing hands the branch a VIEW of
# the box's entity, which is correct exactly when every class the test accepts
# has the target's runtime layout. `isinstance(x, int)` accepts a bool -- Python
# says a bool IS an int -- and a boxed bool is `LyBool_Box`'s three-word
# immortal singleton where an int is not, so the view read `True + 1` as 1.
# `bool` has no entity at all: its runtime shape is `i1`.
#
# ⛔ THE REPAIR IS NOT "narrow to int anyway": that is the silent wrong answer
# the exclusion was put in for. The shape that would work is narrowing
# `isinstance(v, int)` to the UNION `int | bool` -- which the operator dispatch
# added on 2026-09-02 can now add to and index -- but building a union VALUE out
# of an erased box is the reverse of `py.union.unwrap` and does not exist.
# `isinstance(v, bool)` needs an unbox instead of a view, which is a different
# repair again.
def render(value: object) -> str:
    if isinstance(value, str):
        return '"' + value + '"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return "null"


print(render("a"), render(1), render(True), render(None))
