# What: a `type[X]` value whose NAME is also a class in the module. The name
# holds one class and spells another, and the call has to build the one it
# holds. Runtime values, because the two differ only in what they construct --
# a wrong answer here compiles and prints the other class's instance. Every
# other spelling of the same shadowing (an int parameter, a local, a loop
# target) has always worked; none of them reaches the constructor path.


class A:
    def __init__(self, n: int) -> None:
        self.n: int = n
        self.tag: str = "A"


class B:
    def __init__(self, n: int) -> None:
        self.n: int = n * 100
        self.tag: str = "B"


def build(A: type[B], n: int) -> B:
    return A(n)


def build_either(B: type[A], n: int) -> A:
    return B(n)


class Registry:
    def __init__(self, A: type[B]) -> None:
        self.cls: type[B] = A

    def make(self, n: int) -> B:
        return self.cls(n)


def shadowed_local(n: int) -> str:
    A = B
    return A(n).tag


def nested_def(n: int) -> int:
    def A(v: int) -> int:
        return v * 10

    return A(n)


print(build(B, 1).tag, build(B, 1).n)
print(build_either(A, 2).tag, build_either(A, 2).n)
print(Registry(B).make(3).tag)
print(shadowed_local(4))
print(nested_def(4))
print(A(5).tag, B(5).tag)
