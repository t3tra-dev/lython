# An UNANNOTATED method's result type is inferred by walking its body, and the
# body reads fields: `def peek(self): return self.n`. The walk ran before this
# class's fields reached the protocol table, so `self.n` resolved to nothing
# and every such method returned `builtins.object` -- `b.peek() + 1` was then
# refused as "builtins.object does not provide manifest method '__add__'"
# while `b.n + 1` on the next line compiled.
class Bag:
    def __init__(self) -> None:
        self.n = 2
        self.label = "bag"
        self.items = [1, 2, 3]

    def peek(self):
        return self.n

    def name(self):
        return self.label

    def all(self):
        return self.items

    def scaled(self, k: int):
        return self.n * k


b = Bag()
print(b.peek() + 1)
print(b.name() + "!")
print(len(b.all()))
print(b.scaled(3) + 1)


# Inherited fields resolve the same way: the merged field list is what the
# table learns, so a base's field is visible to a subclass's method walk.
class Base:
    def __init__(self) -> None:
        self.base = 10


class Derived(Base):
    def __init__(self) -> None:
        super().__init__()
        self.extra = 5

    def total(self):
        return self.base + self.extra


print(Derived().total() * 2)


# A field whose type comes from another method's inferred result: the walk of
# `twice` reads `self.n` through `peek`, which is itself unannotated.
class Chain:
    def __init__(self) -> None:
        self.n = 7

    def peek(self):
        return self.n

    def twice(self):
        return self.peek() * 2


print(Chain().twice() - 4)
