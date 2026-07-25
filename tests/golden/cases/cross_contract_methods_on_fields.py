# Cross-track: kernel/side-defects implemented list.pop / list.insert and
# tuple.__add__ / __mul__ / count / index, which had no lowering behind their
# declared names, and kernel/4a moved every class field into a box16 heap slot.
# Both branches' own cases use module-level receivers, so neither exercises a
# contract method whose receiver is loaded out of a field slot -- through a
# method, through a parameter, down a nested chain, and out of a container
# element.
#
# `self.xs.insert(...)` is NOT here: insert still requires a rebindable local
# receiver, so the field form is a refusal, pinned by
# errors/list_insert_on_field. The local-alias form the diagnostic recommends is
# below, and it is the one that has to keep working.


class Inner:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs


class Mid:
    def __init__(self, i: Inner) -> None:
        self.i: Inner = i


class Top:
    def __init__(self, m: Mid) -> None:
        self.m: Mid = m

    def drop(self) -> int:
        return self.m.i.xs.pop()

    def drop_at(self, at: int) -> int:
        return self.m.i.xs.pop(at)


t = Top(Mid(Inner([3, 1, 2, 1])))
print(t.m.i.xs)
print(t.drop(), t.m.i.xs)
print(t.drop_at(0), t.m.i.xs)
print(t.m.i.xs.pop(-1), t.m.i.xs)


class Bag:
    def __init__(self, xs: list[int], ts: tuple[int, int, int]) -> None:
        self.xs: list[int] = xs
        self.ts: tuple[int, int, int] = ts

    def take(self) -> int:
        return self.xs.pop()


def take_through_param(b: Bag) -> int:
    return b.xs.pop(0)


b = Bag([3, 1, 2, 1], (5, 6, 5))
print(b.take(), b.xs)
print(take_through_param(b), b.xs)
print(b.ts.count(5), b.ts.index(6), b.ts.index(5))
print(b.ts + (1,), b.ts * 2)


# The workaround errors/list_insert_on_field names: read the slot into a local,
# insert into the local, store it back. This is the alias-read shape that was a
# use-after-free before the field store moved into the slot.
def grow_through_local(bag: Bag, at: int, v: int) -> None:
    local: list[int] = bag.xs
    local.insert(at, v)
    bag.xs = local


grow_through_local(b, 0, 9)
grow_through_local(b, -1, 8)
print(b.xs)

bags = [Bag([1, 2], (4, 4, 9))]
print(bags[0].xs.pop(), bags[0].xs)
print(bags[0].ts.count(4), bags[0].ts.index(9), bags[0].ts * 2)
