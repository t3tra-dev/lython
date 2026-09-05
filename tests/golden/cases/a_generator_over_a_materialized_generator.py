# What: a generator that iterates ANOTHER generator's values. The inner
# generator's frame cannot cross the outer one's suspension, so the values are
# materialized first -- and that spelling has to WORK, because it is the one
# the refusal for the un-materialized shape tells the reader to use. Runtime
# values, because the question is whether the outer generator resumes with the
# inner one's elements intact.

from typing import Iterator


def inner(n: int) -> Iterator[int]:
    for i in range(n):
        yield i * i


def doubled(n: int) -> Iterator[int]:
    for v in list(inner(n)):
        yield v * 2


def paired(n: int) -> Iterator[str]:
    for v in list(inner(n)):
        for w in list(inner(2)):
            yield str(v) + ":" + str(w)


def filtered(n: int) -> Iterator[int]:
    for v in list(inner(n)):
        if v % 2 == 0:
            yield v


print(list(doubled(4)))
print(list(paired(2)))
print(list(filtered(5)))
