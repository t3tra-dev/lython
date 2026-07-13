xs: list[int] = [10, 20, 30]
for x in xs:
    print(x)

print(len(xs))

names: list[str] = ["alice", "bob"]
for name in names:
    print("hi " + name)

words: list[str] = ["x", "y", "z"]
for w in words:
    if w == "y":
        continue
    if w == "z":
        break
    print("kept " + w)


def upto(n: int) -> None:
    for i in range(n):
        print(i)


upto(3)

nums: list[int] = [1, 2, 3]
for a in nums:
    for b in nums:
        print(a * b)
