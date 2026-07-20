# Runtime tuples (tuple(xs), str.partition results, eg.exceptions) iterate
# through the same hoisted position cell the runtime list path uses.

xs = [1, 2, 3]
t = tuple(xs)
for e in t:
    print(e)

s = "a-b-c"
parts = s.partition("-")
for p in parts:
    print(p)

try:
    raise ExceptionGroup("g", [ValueError("v1"), ValueError("v2")])
except* ValueError as eg:
    for e in eg.exceptions:
        print("ve:", str(e))

total = 0
for v in tuple([5, 6, 7]):
    total = total + v
print(total)
