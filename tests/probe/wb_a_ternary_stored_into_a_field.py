# `self.f = <fresh> if c else <borrowed>` is refused by the ownership walk:
#
#     ly.ownership.owned_local_object marks a value this frame never acquired
#
# The same ternary RETURNED compiles (golden:
# an_empty_arm_takes_the_other_ones_element) and leaks nothing over 200 trips;
# the if STATEMENT spelling of the store compiles too (golden:
# a_field_assigned_in_a_branch). What is left is the store.
#
# ⭐ THE MERGE CARRIES ONE MARKER FOR TWO KINDS OF VALUE. One arm is a fresh
# allocation the frame owns; the other is the parameter, which the CALLER
# still holds. Storing into a field TRANSFERS, so the borrowed arm needs a
# retain and the fresh one does not -- and the marker that says which is on the
# merged value, where there is only one answer to give.
#
# ⛔ Not the element type, which is a different defect fixed alongside this
# one: joining `[]`'s `list[object]` with the other arm's `list[int]` built a
# union of two lists, and the ownership message was what that surfaced as. With
# the join repaired the type is right and this remains.
#
# The repair is a retain on the BORROWED arm's edge -- the arm knows which kind
# it is, the merge does not. That is the same shape
# wb_return_a_union_through_finally.py's open half needs: a retain placed where
# the borrow enters the merge rather than at the transfer that reads it.
class C:
    def __init__(self, xs: "list[int] | None") -> None:
        self.xs = [] if xs is None else xs


print(C(None).xs, C([1]).xs)
