# What: an empty container yielded beside a filled one carries no element type
# of its own, and a generator's frame lane is keyed on the joined yield type --
# so only running it shows the elements the consumer actually received.
def maybe_rows():
    yield []
    yield [1]
    yield [2, 3]


print(list(maybe_rows()))


def per_index(n: int):
    for i in range(n):
        yield [] if i == 0 else [i]


print(list(per_index(3)))


def tables():
    yield {}
    yield {"a": 1}


print([sorted(d.items()) for d in tables()])


def sets_of():
    yield set()
    yield {1, 2}


print([sorted(s) for s in sets_of()])
