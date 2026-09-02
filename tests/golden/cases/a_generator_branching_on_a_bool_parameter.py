# What: a bool has no memref shape, so its frame slot is one word holding the
# bit -- and only running both truth values shows the bit that came back out is
# the one that went in, on the first resume and on every later one.
def choose(flag: bool):
    if flag:
        yield 1
    else:
        yield 2


print(list(choose(True)), list(choose(False)))


def guarded(flag: bool):
    if flag:
        yield 1
    yield 2


print(list(guarded(True)), list(guarded(False)))


def each(n: int, flag: bool):
    for i in range(n):
        if flag:
            yield i
        else:
            yield -i


print(list(each(3, True)), list(each(3, False)))


def ternary(flag: bool):
    yield 1 if flag else 2
    yield 3 if flag else 4


print(list(ternary(True)), list(ternary(False)))


def two_flags(a: bool, b: bool):
    for _ in range(2):
        if a:
            yield 1
        if b:
            yield 2


print(list(two_flags(True, False)), list(two_flags(False, True)))


def with_a_string(flag: bool, text: str):
    for ch in text:
        if flag:
            yield ch
        else:
            yield ch.upper()


print(list(with_a_string(True, "ab")), list(with_a_string(False, "ab")))
