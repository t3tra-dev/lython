# A user-defined generic CONTAINER: the specialization has to carry the type
# argument through a container-typed field, the method signatures that read and
# write it, and a self-referential return annotation -- all under two
# instantiations at once.


class Stack[T]:
    def __init__(self) -> None:
        # A dict-backed store, keyed by position: the type argument reaches
        # the field through the annotation's own type parameter.
        self.items: dict[int, T] = {}
        self.size = 0

    def push(self, item: T) -> None:
        self.items[self.size] = item
        self.size = self.size + 1

    def pop(self) -> T:
        self.size = self.size - 1
        item = self.items[self.size]
        del self.items[self.size]
        return item

    def peek(self) -> T:
        return self.items[self.size - 1]

    def __len__(self) -> int:
        return self.size

    def __contains__(self, item: T) -> bool:
        index = 0
        while index < self.size:
            if self.items[index] == item:
                return True
            index = index + 1
        return False

    def to_list(self) -> list[T]:
        out: list[T] = []
        index = 0
        while index < self.size:
            out.append(self.items[index])
            index = index + 1
        return out

    def copy(self) -> Stack[T]:
        clone = Stack[T]()
        index = 0
        while index < self.size:
            clone.push(self.items[index])
            index = index + 1
        return clone


numbers: Stack[int] = Stack()
numbers.push(1)
numbers.push(2)
numbers.push(3)
print(len(numbers), numbers.to_list())
print(numbers.peek())
print(numbers.pop(), numbers.to_list())
print(2 in numbers, 3 in numbers)

clone = numbers.copy()
clone.push(9)
print(numbers.to_list(), clone.to_list())

words: Stack[str] = Stack()
words.push("a")
words.push("b")
print(len(words), words.to_list(), words.pop())
print("a" in words, "b" in words)


# Two type parameters over two container fields.
class Index[K, V]:
    def __init__(self) -> None:
        self.forward: dict[K, V] = {}
        self.count = 0

    def add(self, key: K, value: V) -> None:
        self.forward[key] = value
        self.count = self.count + 1

    def lookup(self, key: K) -> V:
        return self.forward[key]

    def keys(self) -> list[K]:
        out: list[K] = []
        for key in self.forward:
            out.append(key)
        return out


by_name: Index[str, int] = Index()
by_name.add("one", 1)
by_name.add("two", 2)
print(by_name.keys(), by_name.lookup("two"), by_name.count)

by_number: Index[int, str] = Index()
by_number.add(3, "three")
print(by_number.keys(), by_number.lookup(3), by_number.count)
