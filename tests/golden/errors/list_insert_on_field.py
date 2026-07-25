# kernel/side-defects implemented `list.insert`, which had no lowering behind
# its declared name, and its lowering needs a receiver it can REBIND: insert
# grows the list, so the primitive returns a re-rooted lane list that has to be
# stored back somewhere. A class field's storage after kernel/4a is a box16
# slot, not a rebindable local, so the field form is refused rather than
# silently inserting into a copy.
#
# `list.pop` on the same field WORKS (cases/cross_contract_methods_on_fields),
# because pop does not grow. So the boundary is "does the mutation reallocate",
# not "is the receiver a field".
#
# Not a narrowing: on ed6a798 this same program said `runtime manifest has no
# builtins.list.insert method`, i.e. insert did not exist in any spelling. What
# changed is which forms work and how precisely the rest are refused.
#
# The workaround the diagnostic names -- bind the field to a local, insert into
# the local, store it back -- is pinned in that same case, and it is a form that
# was a use-after-free before 4a. When stage 4b puts the payload behind the
# handle, growth stops re-rooting anything and this refusal should become
# unnecessary; this file is how that gets noticed.
class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs


b = Bag([3, 1, 2])
b.xs.insert(0, 9)
print(b.xs)
