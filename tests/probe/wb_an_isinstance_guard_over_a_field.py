# OPEN. An `isinstance` guard over a FIELD does not narrow it:
#
#     class Holder:
#         def __init__(self, a: A) -> None: self.a = a
#         def read(self) -> int:
#             if isinstance(self.a, B):
#                 return self.a.n      # 'n' is overridden by a subclass of 'A'
#             return -1
#
# while binding it to a local first -- `v = self.a; if isinstance(v, B)` -- is
# the same program and compiles. That is the "bound to a name first works"
# tell, which is always two paths answering one question.
#
# ⛔ WHY IT IS NOT A ONE-LINE REPAIR, and why the safe direction is the one
# taken. `BranchTypeNarrowing` keys on a NAME, and the narrowing is spent by
# `applyBranchNarrowing` on the value that name holds. A field is not a value
# the branch holds: it is RE-READ at every use, so narrowing it means emitting
# a `py.class.refine` on each read -- and a refine is a LAYOUT VIEW. If
# anything between the guard and the read replaces the field (a store, or a
# call that stores through another reference to the same object), the view is
# of the wrong class and the read is memory-unsafe, not merely wrong.
#
# So the mechanism this needs is member-path narrowing WITH INVALIDATION: a key
# per path, dropped by any store to that path, any store to its root, and any
# call that could reach the object. Nothing in the emitter tracks that today,
# and a narrowing without it would trade a refusal for an unsound view, which
# is the trade this project refuses.
#
# ⛔ A cheaper half exists and was NOT taken either: the refusal could name the
# workaround ("bind it to a local so the guard can narrow it"). Left alone
# because the message is shared by every unresolvable-dispatch site and only
# this one has that workaround, so the hint would be wrong more often than
# right.
#
# ⭐ AND `is None` OVER A FIELD IS THE SAME GAP, which is what makes this the
# top open item rather than a corner: it is the binary-search-tree idiom.
#
#     class Tree:
#         def __init__(self, v: int) -> None:
#             self.left = None
#         def insert(self, v: int) -> None:
#             if self.left is None:
#                 self.left = Tree(v)
#             else:
#                 self.left.insert(v)   # union<Tree, None> has no 'insert'
#
# ⛔ AND THE SOUND REPAIR IS A CHECKED UNWRAP, not a view. A union value is a
# tag plus lanes; narrowing it means asserting the tag. For a LOCAL the value
# is read once and the assertion holds for the rest of the branch. A field is
# re-read at every use, and between the guard and a later read a call may have
# stored None into it -- unwrapping that as `Tree` yields a garbage pointer,
# which is a memory-safety failure and not a wrong answer. So the narrowing
# would have to be a CHECKED unwrap that raises when the tag disagrees, which
# is sound (CPython raises AttributeError there too) and costs a branch per
# read. That is the mechanism; the current refusal is the honest placeholder.
#
# Measured 2026-09-03: the METHOD spelling (`self.a.only()`), the ATTRIBUTE
# spelling (`self.a.n`) and the `is None` spelling are all refused; binding the
# read to a local first compiles and prints CPython's answer in every one.
class A:
    pass


class B(A):
    n = 7

    def only(self) -> int:
        return 1


class Holder:
    def __init__(self, a: A) -> None:
        self.a = a

    def read(self) -> int:
        if isinstance(self.a, B):
            return self.a.n
        return -1


print(Holder(A()).read(), Holder(B()).read())
