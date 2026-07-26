# probe: leak -- control: an owned list local created and dropped each iteration (40000 iterations)
# axes: op=leak-loop iterations=40000
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 320000

def once() -> int:
    xs: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    return len(xs)


total = 0
for _ in range(40000):
    total += once()
print(total)
