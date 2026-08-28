# A tuple with a UNION element, returned with a different member active on each
# path, unpacked twice. Aborts in `Ly_IncRef observed non-positive refcount` on
# the SECOND unpack, after the first has printed the right answer.
#
#     def make(n: int) -> "tuple[str, int | str]":
#         if n == 0:
#             return ("k", 1)
#         return ("k", "z")
#     k, v = make(0)      # prints k 1
#     k2, v2 = make(1)    # Ly_IncRef observed non-positive refcount
#
# ⛔ THE UNION IS REQUIRED and so are BOTH members. `tuple[str, int]` twice is
# fine (ac1), `tuple[str, str]` twice is fine, a BARE `int | str` return with
# both members is fine (ac6), and the same tuple returning the SAME member on
# both paths is fine (ac4). What is left is: the two returns build the tuple's
# element at DIFFERENT physical shapes, and the merge that reconciles them --
# the evidence tier, which widens each candidate into the union
# (`selectEvidenceObjectByMatch`) -- leaves the reference count of the widened
# member wrong.
#
# ⛔ THIS BLOCKS THE UNION STORE, which is otherwise built and was measured
# working for every straight-line and loop shape (a list literal, `append`,
# `d[k] = v`, a slice, a tuple slice). It was REVERTED because it makes this
# crash reachable from programs that were previously refused -- and worse, one
# of them answers instead of crashing:
#
#     d["x"] = v ; d["y"] = v2      # printed ('y', 'y'), CPython says ('y', 'z')
#
# A refusal that becomes a silent wrong answer is the one trade this project
# does not make. The store is ~60 lines in `objectPayloadHandleWords` (select
# the handle words by the tag, computing every member's unconditionally because
# an inactive member's lanes are the immortal placeholder); it goes back in
# once the merge above is right.
def make(n: int) -> "tuple[str, int | str]":
    if n == 0:
        return ("k", 1)
    return ("k", "z")


k, v = make(0)
print(k, v)
k2, v2 = make(1)
print(k2, v2)
