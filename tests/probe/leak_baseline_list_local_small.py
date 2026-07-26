# probe: leak -- control: an owned list local created and dropped each iteration (100 iterations)
# axes: op=leak-loop iterations=100
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 800

def once() -> int:
    xs: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    return len(xs)


total = 0
for _ in range(100):
    total += once()
print(total)
