# An `Optional[T]` local that a nested callable reads and the enclosing body
# REBINDS cannot be compiled:
#
#     def make() -> int:
#         v: Optional[int] = 3
#         def inner() -> int:
#             return 0 if v is None else v
#         v = 4
#         return inner()
#     # cannot adapt runtime bundle  with physical values (i64, memref<2xi64>)
#     # to expected ABI (memref<2xi64>)
#
# MEASURED (2026-09-06, RelWithDebInfo, today's tree) by sweeping every
# annotation kind through the closure shapes. What separates the four:
#
#   capture WITHOUT a rebind ......................... correct (no cell)
#   an optional reassigned inside a `try` ............ correct (the try's own
#                                                      cell path, which the
#                                                      note at that rule calls
#                                                      "a contract-shaped slot")
#   int / str / list / dict / class / callable in a
#     cell, rebound after the nested def ............. correct
#   Optional[int] in a cell, rebound ................. this file
#   Optional[str] in a cell, rebound ................. an EARLIER message,
#                                                      "union<str, None> does
#                                                      not provide ..."
#   an optional captured by a RETURNED closure ....... " has no statically
#                                                      sized entity lane to
#                                                      rebuild a box from, got
#                                                      'i64'"
#
# ⭐ THE ADAPT MESSAGE IS THE FAMILY'S SIGNATURE, and it appeared at THREE
# unrelated sites in one sweep: a union module GLOBAL cell, a union crossing a
# try/finally join, and this. Each time the value has a union's two lanes (an
# i64 tag and the payload's memref) and the storage boundary declares one. The
# write half of the container path solves exactly this -- `objectPayloadHandleWords`
# selects the ACTIVE member's handle words under a tag guard, unconditionally,
# because an inactive member's lanes are the immortal dead placeholder -- so the
# shape of the repair exists; what is missing is that boundary asking for it.
#
# ⛔ AND THE `try` PATH IS THE EVIDENCE IT CAN WORK. It stores an optional as
# ONE box whose empty entity word IS the None, which is what made a linked
# structure expressible. The cell here is the same class; only the operand
# adaptation on the way in differs.
from typing import Optional


def make() -> int:
    v: Optional[int] = 3

    def inner() -> int:
        return 0 if v is None else v

    v = 4
    return inner()


print(make())
