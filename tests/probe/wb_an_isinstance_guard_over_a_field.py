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
# Measured 2026-09-03: both the METHOD spelling (`self.a.only()`) and the
# ATTRIBUTE spelling (`self.a.n`) are refused; the local-binding spelling of
# each compiles and prints CPython's answer.
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
