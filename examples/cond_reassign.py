def guarded_sum(n: int) -> int:
    acc = 0
    for i in range(n):
        if i < 3:
            acc = acc + i
    return acc


def two_arm(n: int) -> int:
    acc = 0
    for i in range(n):
        if i % 2 == 0:
            acc = acc + i
        else:
            acc = acc + 100
    return acc


def str_tag(n: int) -> str:
    tag = "start"
    for i in range(n):
        if i == 2:
            tag = tag + "!" + str(i)
    return tag


def while_guarded(limit: int) -> int:
    acc = 1
    i = 0
    while i < limit:
        if acc < 20:
            acc = acc * 2
        i = i + 1
    return acc


def after_loop(n: int) -> int:
    best = 0
    for i in range(n):
        if i * i > best:
            best = i * i
    extra = best + 1
    return extra


print(guarded_sum(6))
print(two_arm(5))
print(str_tag(4))
print(while_guarded(10))
print(after_loop(4))

label = "a"
if len(label) == 1:
    label = label + "b"
print(label)
