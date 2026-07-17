# Cross-track (Wave 0 integration): the generator protocol drives big-int
# arithmetic. Yielded values stay on the int tier (the resume ABI is i64
# lanes); the consumer promotes past INT64_MAX, including through yield-from
# delegation. Yielding an already-promoted int is pinned by
# errors/generator_bigint_yield.py until the frame release contract lands.

def factors(n: int):
    i = 1
    while i <= n:
        yield i
        i = i + 1


def squares(n: int):
    i = 0
    while i < n:
        yield i * i
        i = i + 1


def inner(n: int):
    i = 0
    while i < n:
        yield i + 1
        i = i + 1


def outer(n: int):
    yield from inner(n)


fact = 1
for k in factors(30):
    fact = fact * k
print(fact)

total = 0
for s in squares(8):
    total = total * 9223372036854775807 + s
print(total)

g = factors(3)
print(next(g) + 10 ** 20)
print(next(g) * 2 ** 64)

p = 1
for v in outer(64):
    p = p * 2
print(p)
