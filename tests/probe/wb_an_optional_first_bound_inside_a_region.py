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
