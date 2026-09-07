# A name whose FIRST binding is an `Optional[T]` and whose binding sits inside a
# region cannot be read after it:
#
#     w: Optional[int] = 3
#     match flag:
#         case 1: v = w
#         case _: v = w
#     return 0 if v is None else v
#     # unresolved name 'v'
#
# MEASURED (2026-09-06, RelWithDebInfo, today's tree). What isolates it:
#
#   int / str / list / class / callable in the same place ... correct (the
#       callable arm was added the same day, and reads exactly like this one)
#   an optional REASSIGNED inside a try ..................... correct -- the
#       try's own carry-out rule takes `T | None`, storing it as ONE box whose
#       empty entity word IS the None
#   an optional captured by a closure and rebound ........... correct
#   an optional FIRST bound inside for / while / try / match  this file
#
# ⛔ AND THE OBVIOUS RELAXATION MIS-EXECUTES AT RUN TIME. Letting an optional
# through the slot rule -- the same one-line change that was right for a
# callable -- turns "unresolved name 'v'" into "Ly_IncRef observed non-positive
# refcount", which is a use-after-free reported by the runtime rather than a
# refusal: the slot's write and the region's own edge each take the box.
# The two rules reach different storage: the try's cell is written and read on
# the statement's own edges, and this slot is read by whatever FOLLOWS the
# region, which is where the empty-entity encoding stops being enough.
#
# ⭐ SO THE SPELLING THAT LOOKS IDENTICAL IS NOT: `v` bound before the region
# and reassigned inside it compiles, because then there is no slot at all.
# That is also the workaround, and it is one line.
# ⭐ RE-MEASURED 2026-09-08, after a union FIELD became one box. Two of the
# three reasons above have moved:
#
#   letting a union through `inferConditionalLocalType` no longer crashes the
#     compiler: `int | str` first bound inside a for/while/match COMPILES AND
#     RUNS, and `int | None` gets as far as "Ly_IncRef observed non-positive
#     refcount" at run time;
#   routing the optional field through the GENERAL union path -- the class word
#     as the tag rather than the empty entity -- makes those three run
#     correctly too, and `ctest -L fast` stays green.
#
# ⛔ AND THE SLOT RELAXATION IS STILL REFUSED, for a reason that is not about
# storage at all: with it, `golden.cases.stdlib_bisect`, `stdlib_functools`
# and `generic_call_with_a_lambda` fail to EMIT -- a name those modules bind
# inside a region gets an `int | None` slot where the dominance path had given
# it the exact `int`, and `lo + 1` on a union is refused. The slot's TYPE is
# the open question, not the slot's storage: it joins every binding in the
# region, and a None-then-int sequence joins to a union the reader never sees.
#
# ⛔ Removing the optional fast path on its own FIXES NO PROGRAM (measured:
# identical probe and golden differentials, 741/715 either way), so it is not
# committed. It is the second half of this repair and belongs with the first.
from typing import Optional


def use(flag: int) -> int:
    w: Optional[int] = 3
    match flag:
        case 1:
            v = w
        case _:
            v = w
    return 0 if v is None else v


print(use(1))
