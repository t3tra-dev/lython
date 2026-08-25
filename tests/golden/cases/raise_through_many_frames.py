# The personality remembers what it found for a return address, in a table
# inside the exception carrier, and a carrier handed back by a catch keeps it.
# Every way that memo can be looked up wrongly needs a run: the same call site
# at many depths (one slot, many frames), two call sites that land in the same
# slot, and a carrier reused by a later raise that must not answer for the
# earlier one's frames.


def deep(n: int) -> int:
    if n == 0:
        raise ValueError("bottom")
    return deep(n - 1)


def first_site() -> str:
    try:
        raise KeyError("one")
    except KeyError as e:
        return "one " + str(e)


def second_site() -> str:
    try:
        raise IndexError("two")
    except IndexError as e:
        return "two " + str(e)


def alternating(n: int) -> str:
    seen = ""
    for i in range(n):
        if i % 2 == 0:
            seen = first_site()
        else:
            seen = second_site()
    return seen


def caught_at(depth: int) -> str:
    try:
        deep(depth)
    except ValueError as e:
        return "depth " + str(depth) + " " + str(e)
    return "unreachable"


for d in [1, 2, 3, 12, 40]:
    print(caught_at(d))
print(alternating(9))
print(caught_at(3))


def mixed(n: int) -> int:
    total = 0
    for i in range(n):
        try:
            deep(i % 5)
        except ValueError:
            total += 1
        try:
            raise TypeError("t")
        except TypeError:
            total += 10
    return total


print(mixed(7))
