def windowed(n: int):
    i = 0
    while i < n:
        items: list[int] = []
        items.append(i)
        items.append(i * 2)
        yield items[0] + items[1]
        i = i + 1

for v in windowed(3):
    print(v)
print("done")
