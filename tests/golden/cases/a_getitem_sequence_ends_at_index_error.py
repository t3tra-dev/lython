# Why execution: the point is how many elements a loop sees, which only the
# printed values show; every program here compiles either way.
#
# A class with `__getitem__` and no `__iter__` iterates until `__getitem__`
# raises IndexError (or StopIteration) -- `__len__` is never asked. The loop
# was bounded by `len()`, so `Odd`, whose two disagree, printed two elements
# where CPython prints four, and a class with `__getitem__` alone could not
# be iterated. Covered: the for statement (break, continue, else, a tuple
# target, a base-typed value), comprehensions, list/tuple/sum/max/sorted/set,
# `in`, enumerate/zip/map/iter in a loop and as values, and reversed (which
# DOES ask `len`, in CPython too).
#
# And a name bound inside a try in a loop, read after it into a typed list:
# `out.append((i, v))` was refused as a tuple of the name's storage slot.
class Odd:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        if index >= 4:
            raise IndexError("end")
        return index * 10


class OnlyGet:
    def __getitem__(self, index: int) -> str:
        if index >= 3:
            raise IndexError(index)
        return "abc"[index]


class StopsByStop:
    def __getitem__(self, index: int) -> int:
        if index == 2:
            raise StopIteration
        return index


class Pairs:
    def __getitem__(self, index: int) -> tuple[int, str]:
        if index >= 3:
            raise IndexError
        return (index, "p" + str(index))


class Base:
    def __getitem__(self, index: int) -> int:
        if index >= 2:
            raise IndexError
        return 1


class Longer(Base):
    def __getitem__(self, index: int) -> int:
        if index >= 4:
            raise IndexError
        return 2


for v in Odd():
    if v == 10:
        continue
    print("v", v)
else:
    print("else ran")
for v in Odd():
    if v == 20:
        break
    print("b", v)
else:
    print("not printed")
for c in OnlyGet():
    print(c, end="")
print()
for n in StopsByStop():
    print("s", n)
for i, name in Pairs():
    print(i, name)
seqs: list[Base] = [Base(), Longer()]
for seq in seqs:
    print([x for x in seq])
print([x + 1 for x in Odd()], len(Odd()))
total = 0
for v in Odd():
    total += v
print(total)

class G:
    def __getitem__(self, i: int) -> int:
        if i > 2:
            raise IndexError
        return 5 - i



print(list(Odd()), tuple(Odd()), sum(Odd()), max(Odd()), sorted(Odd()))
print(list(G()), tuple(G()), sum(G()), max(G()), sorted(G()), set(G()))
print(30 in Odd(), 4 in G(), 9 in G())
print(list(zip(G(), Odd())), list(zip([1, 2, 3, 4, 5], Odd())))
print(list(enumerate(Odd())), list(map(lambda v: v + 1, G())))
e = enumerate(G())
print(list(e))
z = zip([1, 2, 3, 4, 5], Odd())
print(list(z))
print(list(reversed(Odd())), list(iter(Odd())))
for a, b in zip([1, 2, 3, 4, 5], Odd()):
    print(a, b)


class Counter:
    def __init__(self) -> None:
        self.i = 0

    def __iter__(self) -> "Counter":
        return self

    def __next__(self) -> int:
        if self.i >= 3:
            raise StopIteration
        self.i += 1
        return self.i


def numbered(r: Counter) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    for v in r:
        out.append((i, v))
        i += 1
    return out


print(numbered(Counter()), list(enumerate(Counter())))
