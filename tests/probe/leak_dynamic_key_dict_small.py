# probe: leak -- a literal dict stored into through a computed key (100 iterations)
# axes: op=leak-loop iterations=100
# CPython 3.14 expects: 200

def once(n: int) -> int:
    d: dict[str, list[int]] = {"k": [0] * 200}
    d[str(n % 1)] = [1] * 200
    return len(d)


total = 0
for i in range(100):
    total += once(i)
print(total)
