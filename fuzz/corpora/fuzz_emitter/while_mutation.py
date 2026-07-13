xs: list[int] = []
i = 0
while i < 5:
    xs.append(i * i)
    i += 1

acc = 0
for v in xs:
    if v == 9:
        break
    acc += v
print(xs, acc)
