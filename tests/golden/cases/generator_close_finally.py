def with_cleanup():
    try:
        yield 1
        yield 2
    finally:
        print("cleanup")

g = with_cleanup()
print(next(g))
g.close()
g.close()
print("closed twice ok")

fresh = with_cleanup()
fresh.close()
print("fresh close ok")

def swallows():
    try:
        yield 1
    except GeneratorExit:
        print("swallowed")

s = swallows()
print(next(s))
s.close()
print("swallow close ok")

def ignores():
    while True:
        try:
            yield 1
        except GeneratorExit:
            yield 2

i = ignores()
print(next(i))
try:
    i.close()
except RuntimeError as e:
    print("RuntimeError:", e)

def drains():
    try:
        yield 1
        yield 2
    finally:
        print("drained")

for v in drains():
    print(v)
print("done")
