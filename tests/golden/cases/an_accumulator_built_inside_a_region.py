# What: the empty accumulator declared inside a loop, a try or a with, and the
# ways Python fills one that are not `append`. Every line here decodes an
# element -- adds to it, indexes it, sums it -- because an erased element type
# compiles right up to the first read and prints nothing wrong until then.
def in_a_loop() -> int:
    for _ in range(1):
        xs = []
    xs.append(1)
    xs.append(2)
    return xs[0] + xs[1]


def in_a_try() -> int:
    try:
        counts = {}
    finally:
        pass
    counts["a"] = 3
    return counts["a"] + 1


def in_a_while() -> int:
    i = 0
    while i < 1:
        xs = []
        i += 1
    xs.extend([4, 5])
    return sum(xs)


def either_branch(flag: bool) -> int:
    for _ in range(1):
        if flag:
            xs = []
        else:
            xs = [6]
    xs.append(7)
    return xs[0] + xs[-1]


print(in_a_loop(), in_a_try(), in_a_while(), either_branch(True),
      either_branch(False))


# The filling operations, each on a container nothing else typed.
def by_extend() -> int:
    xs = []
    xs.extend([1, 2])
    return xs[1] + 1


def by_insert() -> int:
    xs = []
    xs.insert(0, 8)
    return xs[0] + 1


def by_augmented() -> int:
    xs = []
    xs += [9]
    return xs[0] + 1


def by_update() -> int:
    d = {}
    d.update({"k": 10})
    return d["k"] + 1


def by_setdefault() -> int:
    d = {}
    d.setdefault("k", 11)
    return d["k"] + 1


def by_dict_or() -> int:
    d = {}
    d |= {"k": 12}
    return d["k"] + 1


print(by_extend(), by_insert(), by_augmented(), by_update(), by_setdefault(),
      by_dict_or())


# An empty container nothing ever fills keeps the slot it always had.
def never_filled() -> None:
    for _ in range(1):
        xs = []
    print(xs, len(xs))


never_filled()
