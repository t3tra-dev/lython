# Generic classes (PEP 695): one class contract per ground instantiation.
# Covers every way an instantiation is determined -- explicit type arguments,
# an annotated context, and the argument types alone -- plus specialization
# reuse, two instantiations of one class living side by side, and inheritance
# from an instantiation.


class Box[T]:
    def __init__(self, value: T) -> None:
        self.value = value

    def get(self) -> T:
        return self.value

    def set(self, value: T) -> None:
        self.value = value


# Explicit type arguments at the call site.
print(Box[int](1).get())
print(Box[str]("two").get())

# The argument types alone determine T.
print(Box(3).get())
print(Box("four").get())
print(Box(5.5).get())
print(Box(True).get())

# An annotated context determines T for an argumentless-looking call.
annotated: Box[int] = Box(6)
annotated.set(7)
print(annotated.get())

# The same instantiation twice reuses one class contract.
first: Box[str] = Box("a")
second: Box[str] = Box("b")
print(first.get(), second.get())


# A generic in a parameter and in a return annotation.
def unwrap(box: Box[int]) -> int:
    return box.get()


def rewrap(value: str) -> Box[str]:
    return Box(value)


print(unwrap(Box(8)))
print(rewrap("nine").get())


# Two type parameters, and two instantiations of the same class.
class Pair[K, V]:
    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value

    def show(self) -> str:
        return str(self.key) + "=" + str(self.value)

    def swapped(self) -> str:
        return str(self.value) + "=" + str(self.key)


print(Pair("k", 10).show())
print(Pair(11, "v").show())
print(Pair("k", 12).swapped())


# A non-generic subclass of an instantiation, and a generic subclass of a
# generic class: the base list linearizes the specialized contract.
class IntBox(Box[int]):
    def doubled(self) -> int:
        return self.get() * 2


counted = IntBox(13)
print(counted.get(), counted.doubled())


class Labeled[T](Box[T]):
    def __init__(self, value: T, label: str) -> None:
        self.value = value
        self.label = label

    def show(self) -> str:
        return self.label + ":" + str(self.get())


labeled: Labeled[int] = Labeled(14, "n")
print(labeled.show())
text: Labeled[str] = Labeled("x", "s")
print(text.show())
