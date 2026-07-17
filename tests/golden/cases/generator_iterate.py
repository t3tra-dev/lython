def counter(n: int):
    i = 0
    while i < n:
        yield i * i
        i = i + 1

def alternating(n: int):
    i = 0
    while i < n:
        if i % 2 == 0:
            yield i
        else:
            yield -i
        i = i + 1

for v in counter(4):
    print(v)

for v in alternating(5):
    print(v)

g = counter(3)
print(next(g))
print(next(g))
print(next(g))
try:
    next(g)
except StopIteration:
    print("exhausted")
try:
    next(g)
except StopIteration:
    print("still exhausted")
