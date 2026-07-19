def run() -> None:
    calls = 0

    def bump(x: int) -> int:
        nonlocal calls
        calls += 1
        return -x

    data = [5, 2, 8, 1, 9]
    data.sort(key=lambda v: bump(v))
    print(data, calls)
    odds: list[int] = []
    for v in map(lambda v: v * 10, filter(lambda v: v % 2 == 1, data)):
        odds.append(v)
    print(odds)
    picked = sorted(odds, key=lambda v: bump(v))
    print(picked)
    print("key calls:", calls)
    total = 0
    for i, v in enumerate(picked):
        total += i * v
    print("weighted:", total)


run()
