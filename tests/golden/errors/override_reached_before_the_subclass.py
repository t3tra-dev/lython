# The guard answered from maps the class emission fills as it goes, so a call
# emitted ABOVE the subclass saw no subclass and inlined the base's body --
# moving `class B` up flipped the same program to a refusal. Whether a
# hierarchy has an override is a property of the module, not of where in it
# the question is asked, so the hierarchy is recorded before anything is
# emitted.
class A:
    def v(self) -> int:
        return 1


def call_it(a: A) -> int:
    return a.v()


class B(A):
    def v(self) -> int:
        return 2


print(call_it(B()))
