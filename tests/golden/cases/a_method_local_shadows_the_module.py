# A method body is emitted INLINE at the call site, so it shared the caller's
# names: a method local whose name also stood at the use site inherited that
# binding, and `for a in xs:` inside a method with `a = "hello"` at module scope
# was refused with "cannot adapt runtime bundle builtins.int ... to expected ABI
# (memref<16xi64>, ...)". Must run: the refusal is what regresses, but only the
# printed values show the local and the global stayed different objects.


class Counterish:
    def __init__(self) -> None:
        self.history: list[int] = []

    def apply(self, amounts: list[int]) -> int:
        for a in amounts:
            self.history.append(a)
        total = 0
        for h in self.history:
            total = total + h
        return total


# `a` is an instance here and an int inside the method. Both keep their own.
a = Counterish()
print(a.apply([10, -3]))
print(a.apply([5]))
print(a.history)

# The use site's name is a str; the method's is an int.
text = "hello"


class Sums:
    def add(self, xs: list[int]) -> int:
        total = 0
        for text in xs:
            total = total + text
        return total


print(Sums().add([1, 2]), text)


# A caller's LOCAL must not reach an inlined body either.
def outer() -> str:
    text = "abc"
    n = Sums().add([3, 4])
    return text + str(n)


print(outer())

# What a method may still do with the module scope: read a global it does not
# bind, mutate a global container, and write one it declares `global`.
limit = 10
shared: list[int] = []
count = 0


class Env:
    def under(self, v: int) -> bool:
        return v < limit

    def push(self, v: int) -> None:
        shared.append(v)

    def bump(self) -> None:
        global count
        count = count + 1


e = Env()
print(e.under(3), e.under(20))
e.push(1)
e.push(2)
print(shared)
e.bump()
e.bump()
print(count)
