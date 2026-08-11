# This used to COMPILE and print 1: inlining answered the call from the
# receiver's static class, so the subclass's body never ran. There is no
# dynamic dispatch to fall back to, so the call is refused where the
# hierarchy is visible rather than silently running the base's body.
class A:
    def v(self) -> int:
        return 1


class B(A):
    def v(self) -> int:
        return 2


a: A = B()
print(a.v())
