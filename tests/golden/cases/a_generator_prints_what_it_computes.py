# What: a fused argument -- a comprehension, a reducer, zip, enumerate -- ends
# the block its call's callee was emitted in, and inside a generator the frame
# lane for that callee is what used to fail. Only running it shows the call
# reached print with the value the fusion built.
def straight():
    print(sum([1, 2, 3]))
    print(any([True, False]), all([True, False]))
    print(set([1, 2]) == {1, 2})
    yield 0


print(list(straight()))


def comprehensions():
    print([n * 2 for n in range(3)])
    print(sorted([3, 1, 2]))
    print(sorted([3, 1, 2], key=lambda n: -n))
    yield 1


print(list(comprehensions()))


def lazies():
    print(list(zip([1, 2], "ab")))
    print([(i, c) for i, c in enumerate("ab")])
    yield 2


print(list(lazies()))


def after_the_yield(n: int):
    yield n
    print(sum([n, n]))


print(list(after_the_yield(3)))


def in_a_loop(n: int):
    for i in range(n):
        print(sum([i, i]))
        yield i


print(list(in_a_loop(2)))
