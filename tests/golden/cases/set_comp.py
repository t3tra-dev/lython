s = {x % 3 for x in range(10)}
print(len(s))
print(sum(s))
print(2 in s)
print(5 in s)

words = {w + "!" for w in "aabca"}
print(len(words))
print("a!" in words)
print("d!" in words)

total = 0
for v in {k * k for k in range(4)}:
    total = total + v
print(total)
def uniques(xs: list[int]) -> int:
    seen = {v for v in xs}
    return len(seen)


print(uniques([3, 1, 3, 2, 1]))


def evens(n: int) -> int:
    pick = {i for i in range(n) if i % 2 == 0}
    return sum(pick)


print(evens(10))

empty = {z for z in range(0)}
print(len(empty))
print(7 in empty)

grow = {g % 7 for g in range(200)}
print(len(grow))
print(sum(grow))

mixed_src = [5, 5, 5, 5]
ded = {m for m in mixed_src}
print(len(ded))
for only in ded:
    print(only)
