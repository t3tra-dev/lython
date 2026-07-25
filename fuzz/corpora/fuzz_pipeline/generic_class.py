class Box[T]:
    def __init__(self, value: T) -> None:
        self.value = value

    def get(self) -> T:
        return self.value

    def copy(self) -> Box[T]:
        return Box[T](self.value)


class Pair[K, V]:
    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value


class Sub(Box[int]):
    def twice(self) -> int:
        return self.get() * 2


class Deep[T](Box[T]):
    def __init__(self, value: T, depth: int) -> None:
        self.value = value
        self.depth = depth


print(Box[int](1).get())
print(Box("s").get())
annotated: Box[float] = Box(2.5)
print(annotated.copy().get())
print(Pair(1, "a").key, Pair("b", 2).value)
print(Sub(3).twice())
deep: Deep[str] = Deep("d", 1)
print(deep.get(), deep.depth)
