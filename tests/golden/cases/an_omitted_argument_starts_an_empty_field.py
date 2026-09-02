# What: the None-default-then-empty-container constructor, which is how every
# Python class avoids sharing one mutable default. Each instance has to get its
# OWN container, and the elements have to come back as what they are -- summing
# them and adding to one is the decode: an erased element would not add, and a
# shared default would make the second instance report the first one's items.
class Bag:
    def __init__(self, xs: "list[int] | None" = None) -> None:
        if xs is None:
            xs = []
        self.xs = xs

    def total(self) -> int:
        return sum(self.xs)


first = Bag()
second = Bag()
first.xs.append(2)
first.xs.append(3)
print(first.xs, second.xs)
print(first.total() + 1, second.total() + 1)
print(Bag([7, 8]).total())


class Index:
    def __init__(self, table: "dict[str, int] | None" = None) -> None:
        if table is None:
            table = {}
        self.table = table

    def bump(self, key: str) -> int:
        self.table[key] = self.table.get(key, 0) + 1
        return self.table[key]


one = Index()
print(one.bump("a"), one.bump("a"), Index({"a": 5}).bump("a"))
print(one.table, Index().table)


# The rebinding only takes an element type the name already carries: an int
# parameter that a branch defaults keeps answering as an int.
class Counter:
    def __init__(self, start: "int | None" = None) -> None:
        if start is None:
            start = 0
        self.start = start


print(Counter().start + 1, Counter(4).start + 1)
