def resilient():
    try:
        yield 1
    except ValueError as e:
        print("body caught:", e)
        yield 99
    yield 2

g = resilient()
print(next(g))
print(g.throw(ValueError("boom")))
print(next(g))

def plain():
    yield 1
    yield 2

h = plain()
print(next(h))
try:
    h.throw(ValueError("unhandled"))
except ValueError as e:
    print("caller caught:", e)
try:
    next(h)
except StopIteration:
    print("closed after throw")
