# WHAT: a comprehension or generator expression whose target is spelled like a
# binding of the enclosing scope -- a module global, a function local, a
# parameter -- leaves that binding alone, in every form: list, set and dict
# comprehensions, a generator expression consumed by sorted/sum/max or a for
# loop, nested comprehensions that reuse the name, and a lambda built inside
# one.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect printed a wrong
# value. Every form is emitted as a loop in the enclosing frame, so the target
# WAS the enclosing binding: `k: int = 5` printed 2 after `[k * 2 for k in
# range(3)]`, where CPython keeps 5. The other shapes were refused for the
# same reason (a str stored where int is declared; one loop-carried local of
# two types; a module name no longer bound once and unreadable from a
# function).
k: int = 5
print([k * 2 for k in range(3)])
print(k)
counts = {"a": 1}
print(sorted((k, n) for k, n in counts.items()))
print({k: 1 for k in "xy"}, sorted({k for k in "yx"}), sum(1 for k in "xy"))
for item in (k for k in "ab"):
    print(item)
print(k + 1)


def doubled() -> int:
    return k * 2


print(doubled())
x = [1, 2, 3]
print([x for x in x])
rows = [[1, 2], [3]]
print([[x * 2 for x in x] for x in rows])
fs = [lambda: x for x in range(3)]
gs = [lambda x=x: x for x in range(3)]
print([f() for f in fs], [g() for g in gs], x)


def local(n: int) -> list[int]:
    k = 10
    vals = [k + k for k in range(n)]
    total = sum(k * j for j in range(k) if j < 2)
    return vals + [k, total, max(n for n in range(n))]


print(local(3))
