# A generator whose yields are inside a guard. The walk that types the yields
# sees each `yield` on its own, with no flow facts, so the declared element
# type looked like a mismatch for a body whose every yield produces it.


from typing import Iterator


def positives(xs: list[int | None]) -> Iterator[int]:
    for v in xs:
        if v is not None:
            yield v * 2


print(list(positives([1, None, 3])))


def skipped(xs: list[int | None]) -> Iterator[int]:
    for v in xs:
        if v is None:
            continue
        yield v + 1


print(list(skipped([1, None, 3])))


def total(xs: list[int | None]) -> int:
    out = 0
    for v in positives(xs):
        out += v
    return out


print(total([1, None, 3]))
