def total(xs: list[int]) -> int:
    s = 0
    for v in xs:
        s = s + v
    return s


def keys(d: dict[str, int]) -> int:
    return len(d)


def tail(xs: list[int]) -> list[int]:
    if len(xs) == 0:
        return []
    out: list[int] = []
    for i in range(1, len(xs)):
        out.append(xs[i])
    return out


print(total([]))
print(total([1, 2, 3]))
print(keys({}))
print(total(tail([7, 8, 9])))
