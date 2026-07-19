def squares(n: int):
    i = 1
    while i <= n:
        yield i * i
        i += 1


def run() -> None:
    for idx, v in enumerate(squares(3)):
        print(idx, v)
    for a, b in zip(squares(4), [10, 20, 30]):
        print(a, b)
    it = squares(2)
    print(next(it, -1))
    print(next(it, -1))
    print(next(it, -1))
    total = 0
    for v in filter(lambda q: q % 2 == 0, squares(6)):
        total += v + 1
    print(total)
    collected: list[int] = []
    for _, v in zip(squares(5), [0, 0, 0]):
        collected.append(v + 1)
    z = enumerate(collected, 7)
    pair = next(z)
    print(pair)
    pair = next(z)
    print(pair)
    print(next(z, (-1, -1)))
    print(next(z, (-1, -1)))


run()
