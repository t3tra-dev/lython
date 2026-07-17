pairs = [(1, "a"), (2, "b"), (3, "c")]
out = [k for k, v in pairs]
print(out)
names = [v for k, v in pairs if k > 1]
print(names)
swapped = {v: k for k, v in pairs}
print(len(swapped))
print(swapped["a"])
total = 0
for s in [w for _, w in pairs]:
    total = total + len(s)
print(total)
