# builtins.list is one handle (memref<9xi64>) whose word 4 holds the items
# array's base address, so a reallocation is a write THROUGH the root and every
# holder observes it. This case pins that property in the five shapes a lane
# travelling beside the root could have faked, and each of them needs a growth
# past the initial 64-slot capacity to be a real test.
#
# Shape 1: an element read back after the array moved. A holder keeping the old
# items lane would read the freed block.
xs: list[int] = []
i = 0
while i < 200:
    xs.append(i)
    i = i + 1
print(len(xs), xs[0], xs[63], xs[64], xs[199])

# Shape 2: a second binding to the same list, read after the FIRST binding grew
# it. Both names are the same handle, so both see the new base.
ys = xs
xs.append(200)
print(len(ys), ys[200], ys[0])


# Shape 3: a field slot on a call-produced instance, grown through the field
# chain. The slot holds the handle, so the growth does not have to be written
# back into the slot.
class Box:
    def __init__(self) -> None:
        self.items: list[int] = []


def make() -> Box:
    return Box()


b = make()
j = 0
while j < 130:
    b.items.append(j * 2)
    j = j + 1
print(len(b.items), b.items[0], b.items[64], b.items[129])

# Shape 4: a slice assignment that reallocates, then a read of a surviving
# element. The splice allocates a fresh array and publishes it through the
# handle; a stale lane would name the block it just freed.
zs = [0, 1, 2, 3, 4, 5, 6, 7]
zs[2:6] = list(range(100))
print(len(zs), zs[0], zs[1], zs[2], zs[101], zs[102], zs[103])

# Shape 5: a payload derived from a slice and from a concatenation, both of
# which allocate a fresh list of their own.
part = zs[50:60]
joined = part + [999, 1000]
print(part, len(joined), joined[10], joined[11])

# Shape 6: the mutation AFTER the splice, which is a different question from
# the read after it. A void mutator hands the receiver back unchanged, so
# compile-time element evidence survives the splice unless it is dropped
# explicitly -- and evidence naming the PRE-splice contents answers the read
# above correctly and then mis-executes this delete. That asymmetry is why the
# read is not enough: a case that stopped at the print would have passed.
del zs[0]
print(len(zs), zs[0], zs[1], zs[100], zs[102])
