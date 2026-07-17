def measured():
    yield 1
    return 42

g = measured()
print(next(g))
try:
    next(g)
except StopIteration as e:
    print(e)

def measured_none():
    yield 5

h = measured_none()
print(next(h))
try:
    next(h)
except StopIteration as e:
    print("empty:", e)
