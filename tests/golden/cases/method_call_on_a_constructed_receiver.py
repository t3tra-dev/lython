# Why execution: the values are the assertion. A call through a receiver whose
# class is known exactly must run THAT class's body even when subclasses
# override the method -- and the refusal added for the unresolvable case must
# not reach any of these. Every shape here was silently correct before and has
# to stay correct; the errors golden holds the ones that used to be silently
# WRONG.


class A:
    def v(self) -> int:
        return 1


class B(A):
    def v(self) -> int:
        return 2


class C(B):
    pass


class Plain:
    def w(self) -> int:
        return 5


class Derived(Plain):
    def w(self) -> int:
        return 6

    def both(self) -> int:
        return self.w() + Plain().w()


def from_a_local() -> int:
    x = A()
    return x.v()


def from_a_local_sub() -> int:
    x = B()
    return x.v()


def inside_a_function() -> int:
    x = B()
    return x.v()


def concrete_parameter(b: B) -> int:
    return b.v()


class Holder:
    def __init__(self) -> None:
        self.a = B()

    def go(self) -> int:
        return self.a.v()


def main() -> None:
    print(A().v(), B().v(), C().v())
    print(from_a_local(), from_a_local_sub(), inside_a_function())
    print(concrete_parameter(B()))
    print(Holder().go())
    print(Derived().both())
    for x in [B(), B()]:
        print(x.v())


main()
