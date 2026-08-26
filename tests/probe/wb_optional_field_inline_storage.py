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
# CLOSED 2026-08-27 FOR A MEMBER THAT IS ONE ENTITY; OPEN OTHERWISE
# ============================================================
# All four consequences below were "a union field is stored INLINE, so the
# field's value IS a slice of the instance's SSA lane list". `T | None` is now
# stored in a BOX -- the same box every object-typed field gets, with a zero
# entity for None -- WHEN T is one runtime lane. Three of the four are closed
# for those, and this file's own program is one of them (it was GAP and now
# AGREEs). `int | None` and `str | None` still take the inline path and still
# behave exactly as recorded.
#
# ⛔ THE LIMIT IS ONE LANE, and it is the ABSENT read that sets it, not the
# layout. A box hands back the lanes past the entity through the contract's
# `lane_words` primitive, which DEREFERENCES the entity: `builtins.str` reads
# word 2 of the header. The stand-in for an absent object is the immortal dead
# global, two words wide, so a second lane would be read off the end of it.
# `builtins.list` and `builtins.bytes` are one lane and are boxed;
# `builtins.str` is two and is not. Measured: `class Holder: name: str | None`
# still refuses `got = h.name; print(got)`, and the same program with
# `list[int] | None` prints CPython's answer.
#
# (1) A STORE THROUGH A PARAMETER, CLOSED for a boxed member:
#         def rebind(b: Box) -> None: b.f = fresh   # b.f: list[int] | None
#     lands in the instance and the caller reads it back
#     (wb_param_store_optional_boxed). Still refused for `int | None`
#     (wb_param_store_optional, wb_param_store_readboth_optional): an int has no
#     entity to put in a box, so its optional has nowhere else to live.
#
# (2) A SELF-REFERENTIAL OPTIONAL FIELD, CLOSED. `nxt: Optional["Node"]` -- the
#     shape every linked structure is written in -- has a finite layout now,
#     because a box is a fixed width whatever it points at.
#     tests/golden/cases/self_referential_union_field.py.
#
# (3) READING THE FIELD WITHOUT NARROWING, CLOSED for a boxed member. The read
#     takes its own reference (a retain rooted on the box's payload), so the
#     union's lanes no longer alias the instance's.
#
#     ⛔ AND THE RETAIN'S POSITION IS THE WHOLE REPAIR, not its presence. The
#     release planner puts the instance's death after its LAST USE, and a field
#     read's only use of the instance is the box body -- so the arm has to be
#     decided from a SECOND read of the entity word, taken after the retain, or
#     the instance is deallocated between the lane and the retain. Measured
#     three ways before it was right: a retain inside an `scf.if` arm is not
#     rooted at all ("marks a value this frame never acquired" --
#     `ownedLocalMarkerIsRetainRooted` wants the retain to be the token's
#     IMMEDIATELY preceding op), a retain after the branch runs on freed memory
#     ("Ly_IncRef observed non-positive refcount"), and only the branch-free
#     form with the tag read last places the dealloc after both.
#
# (4) OVERWRITING THE FIELD, CLOSED for a boxed member: the store swaps the
#     box's payload and leaves the instance's lane tuple alone, so the instance
#     keeps its death. tests/probe/leak_optionalfield_rebind_* measures 0 B per
#     iteration over create/set/set/clear. The 41 B and 8264 B figures below
#     stand for the inline members that remain.
#
# The measurements that made the inline case legible, kept because they are
# what the boxed form has to keep beating:
#
#         str            net 1 alloc  /    41 B
#         bytes          net 1 alloc  /    65 B
#         list[int]      net 2 allocs /  8264 B
#         dict[str, int] net 5 allocs / 17001 B
#
#     ⛔ AND THE FIRST READING WAS WRONG, which is why the numbers are here.
#     "The splice releases a one-memref member and not a two-lane one" fitted
#     the first two rows and is false: the IR contains BOTH retains and BOTH
#     releases for the member, correctly paired. What was missing is the call to
#     `__ly_dealloc_H`: the store RE-ROOTED the instance's lane tuple, and the
#     release planner's identity is the tuple, so the instance it was tracking
#     no longer had a death. Boxing the field is exactly what stops the
#     re-rooting.
#
# ⛔ AND ONE MORE, WHICH IS NOT THIS CAUSE and was measured on both binaries:
# storing into one Optional field after a branch narrowed ANOTHER is
# "operand #0 does not dominate this use" (b3 below). Pre-existing, unchanged
# by either repair, and not obviously the same thing -- it is a placement
# problem in the branch, not a storage one.

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
