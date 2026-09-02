# What: a constructor that writes its fields inside a region -- a loop, a
# while, a try -- rather than at the top level. The object has to be released
# by its caller like any other, so this builds many of them and reads what
# each one holds: a missing release is a leak the gate catches, and a marker
# left in the wrong place is a refusal.
class Empty:
    def __init__(self) -> None:
        for _ in range(1):
            self.v = None


class Counted:
    def __init__(self, n: int) -> None:
        i = 0
        while i < 1:
            self.n = n * 2
            i += 1


class Guarded:
    def __init__(self, name: str) -> None:
        try:
            self.name = name.upper()
        finally:
            pass


class Mixed:
    def __init__(self, n: int) -> None:
        for i in range(2):
            self.tag = None
            self.seen = i + n


print(Empty().v, Counted(3).n, Guarded("ab").name)
mixed = Mixed(10)
print(mixed.tag, mixed.seen, mixed.seen + 1)

total = 0
for i in range(5):
    total += Counted(i).n + Mixed(i).seen
print(total)


# A field whose value mentions the LOOP TARGET, which is what such a body
# usually says: the element type has to reach the field declaration.
class Indexed:
    def __init__(self) -> None:
        for i, word in enumerate(["ab", "cd"]):
            self.at = i
            self.word = word


indexed = Indexed()
print(indexed.at + 1, indexed.word.upper())


# A field whose value is a CONTAINER built in the region: reading an element
# back out of it is the decode, because the elements the literal produced live
# in the region and the read does not.
class Paired:
    def __init__(self) -> None:
        for _ in range(1):
            self.pair = (1, "a")
            self.table = {"k": 2}
            self.items = [3, 4]


paired = Paired()
print(paired.pair[0] + 1, paired.pair[1], len(paired.pair))
print(paired.table["k"] + 1, paired.items[0] + paired.items[1])


# The same tuple built at the top level still folds: a read in the block that
# built it keeps the evidence.
class Direct:
    def __init__(self) -> None:
        self.pair = (5, "z")


print(Direct().pair[0] + 1, Direct().pair[1])


# The same field written from a self-read, which must keep re-rooting: the
# lanes it stores are new ones.
class Growing:
    def __init__(self) -> None:
        self.xs = [0]

    def add(self, n: int) -> int:
        ks = self.xs
        ks.append(n)
        self.xs = ks
        return len(self.xs)


g = Growing()
print(g.add(1), g.add(2), g.xs)
