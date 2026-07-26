# An interior view -- a container read out of a field slot into a local -- is
# mutated in a way that REALLOCATES the payload, and the field is then read
# again. What is asserted is that the two agree: the local sees the mutation and
# so does the slot, without the program writing the value back.
#
# Four combinations, because each one reached a different lowering arm and three
# of them used to leave the slot naming storage the mutation had freed:
#
#   dict, one insert            runtime dict setitem, rebind arm
#   list, growth past capacity  runtime list append through ensure_capacity
#   list, growth inside a loop  the same, with the view arriving as a block arg
#   set,  one add               runtime set add
#
# The list cases need more than 64 elements on purpose: a shorter list is
# allocated with capacity 64, so an append does not reallocate and the shape
# cannot distinguish a lowering that publishes the new payload from one that
# never had to.


class Bag:
    def __init__(self, d: dict[str, int], xs: list[int], ys: list[int],
                 s: set[int]) -> None:
        self.d: dict[str, int] = d
        self.xs: list[int] = xs
        self.ys: list[int] = ys
        self.s: set[int] = s


def make() -> Bag:
    d0: dict[str, int] = {"a": 1}
    xs0: list[int] = []
    ys0: list[int] = []
    s0: set[int] = {1}
    i: int = 0
    while i < 65:
        xs0.append(i)
        ys0.append(i)
        i = i + 1
    return Bag(d0, xs0, ys0, s0)


def touch(b: Bag) -> None:
    dv: dict[str, int] = b.d
    dv["z"] = 9
    xv: list[int] = b.xs
    xv.append(999)
    sv: set[int] = b.s
    sv.add(2)


def grow_in_loop(b: Bag) -> None:
    yv: list[int] = b.ys
    i: int = 0
    while i < 40:
        yv.append(i)
        i = i + 1


bag = make()
touch(bag)
grow_in_loop(bag)
print(len(bag.d))
print(len(bag.xs))
print(bag.xs[65])
print(len(bag.ys))
print(bag.ys[100])
print(len(bag.s))

# Same frame, not across a call: the read direction never depended on the
# function boundary, and a fix that only repaired one of the two would show here.
bag2 = make()
d2: dict[str, int] = bag2.d
d2["y"] = 7
x2: list[int] = bag2.xs
x2.append(888)
print(len(bag2.d))
print(len(bag2.xs))
print(bag2.xs[65])
