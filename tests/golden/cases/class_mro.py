class A:
    def who(self) -> str:
        return "A"

    def greet(self) -> str:
        return "hello from " + self.who()


class B(A):
    def who(self) -> str:
        return "B"


class C(A):
    def who(self) -> str:
        return "C"


class D(B, C):
    pass


class Mixin:
    def size(self) -> int:
        return 0

    def describe(self) -> str:
        if self.size() == 0:
            return "empty"
        return "has " + str(self.size()) + " items"


class Box(Mixin):
    def __init__(self, count: int) -> None:
        self.count = count

    def size(self) -> int:
        return self.count


print(A().who())
print(B().who())
print(D().who())
print(D().greet())
print(Box(0).describe())
print(Box(3).describe())
