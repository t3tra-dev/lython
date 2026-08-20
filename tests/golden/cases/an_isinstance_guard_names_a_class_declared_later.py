# `isinstance(a, B)` inside a function defined ABOVE `class B(A)` folded to
# False and ran the else branch: the subtype walk reads the class ops the
# emission creates as it reaches each ClassDef, so a class further down the
# module was, at that point, not a subclass of anything. Silent -- the program
# ran and printed the other branch's answer. Must run: the defect IS the branch
# taken, and both binaries compile this without a diagnostic.


class A:
    def __init__(self, n: int) -> None:
        self.n = n


def kind(a: A) -> str:
    if isinstance(a, B):
        return "b"
    return "a"


def kind_or(a: A) -> str:
    # The same question through the value path rather than the branch path.
    if isinstance(a, B) or a.n > 100:
        return "big-or-b"
    return "plain"


class B(A):
    pass


print(kind(B(1)), kind(A(2)))
print(kind_or(B(1)), kind_or(A(2)), kind_or(A(200)))

# The negative control: a class declared BEFORE the use answered correctly all
# along, and must keep doing so.
class C(A):
    pass


def kind2(a: A) -> str:
    if isinstance(a, C):
        return "c"
    return "a"


print(kind2(C(1)), kind2(A(2)), kind2(B(3)))
