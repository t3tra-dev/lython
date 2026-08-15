# PARTIALLY FIXED 2026-08-16. One of five failures around an Optional FIELD is
# closed; the other four share one cause, which is a mechanism and not a wiring
# error. Everything below was measured on one binary, with the pre-repair
# binary re-run beside it wherever the question is "did this repair cause it".
#
# ⭐ FIXED -- `x.f is None` and a narrowed read, for a member with MORE THAN ONE
# LANE. `lowerAttrGet` consulted the field-evidence cache, which records what
# was last STORED: `self.name: str | None = "a"` caches a `builtins.str`, and
# `isAssignableTo(str, str | None)` is true, so the narrower bundle came back
# where the union's own lanes were expected. Every consumer of a union reads
# the tag from lane 0, so `union.test` handed the str's HEADER MEMREF to
# `arith.cmpi`, whose builder infers the result shape from the operand:
#
#     runtime bundle value 0 for 'builtins.bool' has type 'memref<2xi1>',
#     but ABI expects 'i1'
#
# ⛔ `int | None` reached the same code and SURVIVED, which is why this stood:
# an int member is one lane, so lane 0 of the cached bundle is a memref header
# too but the comparison shape happened to check out further down. The bug is
# the lane, not the width -- a one-lane member compares a header ADDRESS
# against the tag constant 0 or 1, which is a silent wrong answer waiting for
# an allocator that returns a low address.
#
# golden: tests/golden/cases/optional_field_multilane_member.py (red-checked).
#
# ============================================================
# OPEN, all four the same cause: A UNION FIELD IS STORED INLINE
# ============================================================
# `classFieldStoredBoxed` returns false for a union -- "its tag plus every
# member's lanes stay inline, because the box words hold ONE payload handle and
# a union is not one object" -- so the field's value IS a slice of the
# instance's SSA lane list. Four consequences, each with its own diagnostic:
#
# (1) A STORE THROUGH A PARAMETER is refused, because the splice writes the
#     callee's own lanes and the caller holds its own copy:
#         def rebind(b: Box) -> None: b.f = 5     # b.f: int | None
#     "storing into field 'f' of a receiver that arrived as a parameter is not
#     supported for this field's type". `self.f = v` inside a METHOD is fine --
#     the method is inlined at the call site, so the store lands on the
#     caller's value. Two probes: wb_param_store_optional,
#     wb_param_store_readboth_optional.
#
# (2) A SELF-REFERENTIAL OPTIONAL FIELD has no finite layout, so the shape
#     every linked structure is written in is refused outright:
#         class Node:
#             def __init__(self, v: int) -> None:
#                 self.nxt: "Node | None" = None
#     "class layout for 'Node' contains itself through a union-typed field".
#     The diagnostic suggests `nxt: "Node"`, which compiles and cannot express
#     the end of the list.
#
# (3) READING THE FIELD WITHOUT NARROWING loses the instance's release:
#         got = h.name; print(got)
#     "owned resource from builtin.unrealized_conversion_cast result 0 reaches
#     function exit without release". The union's lanes alias the instance's,
#     so a use of the slice is not a use of the group.
#
# (4) OVERWRITING THE FIELD LEAKS THE WHOLE INSTANCE, and this is the one that
#     COMPILES. Measured with `tests/leak_gate.py`, one store over a
#     constructor-set value, every member type:
#
#         str            net 1 alloc  /    41 B
#         bytes          net 1 alloc  /    65 B
#         list[int]      net 2 allocs /  8264 B
#         dict[str, int] net 5 allocs / 17001 B
#
#     ⛔ AND THE FIRST READING WAS WRONG, which is why the numbers are here.
#     "The splice releases a one-memref member and not a two-lane one" fitted
#     the first two rows and is false: the IR contains BOTH retains and BOTH
#     releases for the member, correctly paired
#     (`aggregate_retain = "builtins.str:class.f"` twice,
#     `aggregate_release` for the old slot and for the source). What is missing
#     is the call to `__ly_dealloc_H` -- the deallocator exists, releases the
#     member, and is never reached. The store RE-ROOTS the instance's lane
#     tuple (that is what a splice is), and the release planner's identity is
#     the tuple, so the instance it was tracking no longer has a death. Same
#     cause as (3), and the leak scales with what the instance holds rather
#     than with the field.
#
#     Identical on the binary before the repair above, so it is exposed by that
#     repair rather than caused by it.
#
# ⭐ THE MECHANISM IS THE ONE `lowerAttrSet` ALREADY NAMES: "Refusing is the
# floor until the field is stored behind a handle the way every other field
# is". For `X | None` specifically the box needs NO extra tag word --
# `objectPayloadHandleWords` already writes word 1 = payload class and returns
# an all-zero handle for None, so `box[1] != 0` IS the tag, and the layout
# cycle in (2) disappears because a box is a fixed width. A wider union needs a
# real tag word (word 15 of the box16 is unused) and a switch on both sides.
#
# ⛔ AND ONE MORE, WHICH IS NOT THIS CAUSE and was measured on both binaries:
# storing into one Optional field after a branch narrowed ANOTHER is
# "operand #0 does not dominate this use" (b3 below). Pre-existing, unchanged
# by the repair, and not obviously the same thing -- it is a placement problem
# in the branch, not a storage one.

class Holder:
    def __init__(self) -> None:
        self.items: list[int] | None = None
        self.count: int | None = None


h = Holder()
h.count = 7
n = h.count
if n is not None:
    print(n + 1)
h.items = None
print(h.items is None)
