def leaky():
    yield 1
    raise StopIteration()

g = leaky()
print(next(g))
try:
    next(g)
except RuntimeError as e:
    print("RuntimeError:", e)
try:
    next(g)
except StopIteration:
    print("exhausted after runtime error")
