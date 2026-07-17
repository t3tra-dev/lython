# list values cross the generator suspension ABI on the object-family value
# lane. Each yielded list is built inside the body, transferred to the
# consumer (CPython: the caller of next() receives its own reference), and
# released by the consumer when the loop value dies.
def pairs(n: int):
    i = 0
    while i < n:
        items: list[int] = []
        items.append(i)
        items.append(i * 10)
        yield items
        i = i + 1


for p in pairs(3):
    print(p[0] + p[1])
    print(len(p))
print("done")
