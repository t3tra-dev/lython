# `self.items.append(<ternary>)` is refused:
#
#   list.append on a field or borrowed list is not supported inside a branch or
#   loop body; bind the list to a local variable and mutate the local instead
#
# MEASURED (2026-09-02, RelWithDebInfo, today's tree). The receiver is a field
# in every row; only the ARGUMENT changes:
#
#   c.items.append(1) ............................. correct
#   c.items.append(pick()) ........................ correct
#   c.items.append(v)  (a ternary bound first) .... correct
#   c.items.append(1 if flag else 2) .............. the message above
#   c.items.append(sum(v for v in xs)) ............ the message above
#   c.items.extend([1] if flag else [2]) .......... correct
#   c.table["a"] = 1 if flag else 2 ............... correct
#   c.tags.add(1 if flag else 2) .................. correct
#   the same appends inside a for loop or an if STATEMENT ... correct
#
# ⭐ IT IS THE ARGUMENT'S BLOCKS, not the append's position. A ternary lowers
# to a cf DIAMOND (deliberately -- region results are invisible to the bundle
# machinery), so the receiver `self.items` is read BEFORE the diamond and the
# append lands in the JOIN block. `crossesStorageDefiningBlock` is a same-block
# test, so the two differ and the evidence tier is declined. A `for` loop is an
# scf region, so the read and the append share a block there; an `if` statement
# reads the field again inside each arm.
#
# ⛔ THE TEST IS NOT MERELY CONSERVATIVE HERE. Relaxing it to "the storage
# block dominates" is unsound in general: a mutation in ONE arm of a branch
# leaves the join's evidence right on one path and wrong on the other, and that
# is the silent class this compiler exists to avoid. What the shape needs is a
# runtime append tier for a receiver with no local to rebind -- `set.add` and
# `dict[k] =` on the same field have one, which is why they are in the correct
# column.
class Bag:
    def __init__(self) -> None:
        self.items: "list[int]" = []

    def add(self, flag: bool) -> None:
        self.items.append(1 if flag else 2)


b = Bag()
b.add(True)
b.add(False)
print(b.items)
