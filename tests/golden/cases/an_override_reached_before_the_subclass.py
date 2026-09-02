# What: a call through a base-typed reference emitted ABOVE the subclass that
# overrides the method. The dispatcher tests the runtime class and calls the
# body that class declares, and `B` had no method table until its own ClassDef
# was emitted -- so this was refused ("'B.v' is used before 'B' is defined"),
# with advice to move the class up. Two sibling subclasses that both dispatch
# through the base made that advice impossible to follow: whichever one comes
# second is the one the first cannot see.
#
# WHY THIS IS RUN: which body a dispatch reaches is a runtime fact, and a
# compiler that resolved the base's body instead prints an answer of the same
# shape. The decode is that every arm below prints the SUBCLASS's word, and the
# two-sibling half prints both words from one list.
class A:
    def v(self) -> int:
        return 1


def call_it(a: A) -> int:
    return a.v()


class B(A):
    def v(self) -> int:
        return 2


print(call_it(B()), call_it(A()))


class Expr:
    def show(self) -> str:
        return "?"


class Add(Expr):
    def __init__(self, a: Expr, b: Expr) -> None:
        self.a = a
        self.b = b

    def show(self) -> str:
        return "(" + self.a.show() + "+" + self.b.show() + ")"


class Num(Expr):
    def __init__(self, v: int) -> None:
        self.v = v

    def show(self) -> str:
        return str(self.v)


class Mul(Expr):
    def __init__(self, a: Expr, b: Expr) -> None:
        self.a = a
        self.b = b

    def show(self) -> str:
        return "(" + self.a.show() + "*" + self.b.show() + ")"


tree: Expr = Add(Num(2), Mul(Num(3), Num(4)))
print(tree.show())
nodes: "list[Expr]" = [Num(1), Add(Num(1), Num(2)), Mul(Num(2), Num(5))]
print([n.show() for n in nodes])
