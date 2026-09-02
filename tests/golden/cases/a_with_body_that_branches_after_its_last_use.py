# What: a `with` body with control flow after the last use of its `as` value.
# The normal-path death of the `__enter__` result was written before the
# branch, and an unwind out of any call after it reached the handler's entry
# release of the same value -- so the affine verifier refused the program and
# none of these compiled at all. Using the handle again AFTER the branch moved
# its last use past it and the same program built, which is the same fact read
# from the other side.
#
# WHY THIS IS RUN: what the repair moves is a RELEASE, and a release that runs
# twice or not at all is invisible in the source. The printed values say the
# bodies still ran and produced the numbers CPython produces; the leak gate
# beside this case says the count of releases is right, which no golden can.
#
# ⛔ Every arm here is a different way to split the block after the last use:
# an `if`, an int add (whose fast/slow diamond is an `scf.if` in the same
# block), a floor division, and a `with` inside a loop.
def branched(flag: bool) -> int:
    with open("/dev/null") as handle:
        size = len(handle.read())
        if flag:
            size = size + 1
    return size


print(branched(True), branched(False))


def summed(n: int) -> int:
    total = 5
    with open("/dev/null") as handle:
        total = n + len(handle.read())
    return total


print(summed(3))


def halved() -> int:
    with open("/dev/null") as handle:
        size = len(handle.read())
        total = size // 2
    return total


print(halved())


def looped(xs: "list[int]") -> int:
    total = 0
    for x in xs:
        with open("/dev/null") as handle:
            size = len(handle.read())
            if x > 1:
                total = total + x + size
    return total


print(looped([1, 2, 3]))
