# What: `set()` is the only way to write an empty set, so the None-default
# constructor that gives each instance its own set can only be written as a
# call -- and `list()`, `dict()` and `[]` have to mean the same thing in that
# position. Adding to each container and reading an element back is the decode:
# an erased element neither stores nor comes back as a number.
class Tags:
    def __init__(self, tags: "set[int] | None" = None) -> None:
        if tags is None:
            tags = set()
        self.tags = tags

    def mark(self, n: int) -> int:
        self.tags.add(n)
        return len(self.tags)


first = Tags()
second = Tags()
print(first.mark(3), first.mark(3), sorted(first.tags), sorted(second.tags))
print(sorted(Tags({1, 2}).tags))


class Items:
    def __init__(self, xs: "list[int] | None" = None) -> None:
        if xs is None:
            xs = list()
        self.xs = xs


held = Items()
held.xs.append(4)
print(held.xs, Items().xs, Items([5]).xs, held.xs[0] + 1)


class Counts:
    def __init__(self, d: "dict[str, int] | None" = None) -> None:
        if d is None:
            d = dict()
        self.d = d

    def bump(self, key: str) -> int:
        self.d[key] = self.d.get(key, 0) + 1
        return self.d[key]


one = Counts()
print(one.bump("a"), one.bump("a"), one.d, Counts().d)


# The same rebinding in a plain function, where the branch join is what the
# constructor spelling used to break.
def widen(xs: "list[int] | None" = None) -> int:
    if xs is None:
        xs = list()
    xs.append(1)
    return sum(xs)


print(widen(), widen([10]))
