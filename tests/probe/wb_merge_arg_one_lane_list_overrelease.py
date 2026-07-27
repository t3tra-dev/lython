def run(n: int) -> int:
    total = 0
    for i in range(n):
        xs: list[int] = [i]
        ys: list[int] = xs if i % 2 == 0 else [i, i]
        total += len(ys)
        total += len(xs)
    return total


print(run(6))
