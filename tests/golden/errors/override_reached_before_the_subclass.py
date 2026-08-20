# The guard answered from maps the class emission fills as it goes, so a call
# emitted ABOVE the subclass saw no subclass and inlined the base's body --
# moving `class B` up flipped the same program to a refusal. Whether a
# hierarchy has an override is a property of the module, not of where in it
# the question is asked, so the hierarchy is recorded before anything is
# emitted.
#
# The override itself is now DISPATCHED when the subclass stands above the use
# (tests/golden/cases/a_subclass_body_is_reached_through_the_base.py). What is
# refused here is the ORDER: the dispatcher tests the runtime class and calls
# the body that class declares, and `B` has no method table before its ClassDef
# is emitted. The refusal names that, because the fix is to move the class.
class A:
    def v(self) -> int:
        return 1


def call_it(a: A) -> int:
    return a.v()


class B(A):
    def v(self) -> int:
        return 2


print(call_it(B()))
