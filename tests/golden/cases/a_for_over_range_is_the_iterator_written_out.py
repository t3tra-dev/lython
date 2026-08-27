# WHAT: `for` over `range` -- the element sequence for positive, negative,
# empty and backwards bounds, `continue` / `break` / `else`, a target the body
# rebinds, bounds the body rebinds, a variable step, a zero step, nesting, a
# comprehension, a generator, and a `range` the program binds itself.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the loop is no longer a
# range object and an iterator. It is a counter, a comparison and an add, so
# what a lower layer would see is a `while` with nothing to say whether it
# yields what the iterator did. Only the elements say that.
#
# ⛔ THE `continue` CASE IS THE ONE THE SHAPE IS BUILT AROUND. `continue` in a
# `while` jumps to the test, so an advance written after the body -- which is
# where a `for` body would put it -- is skipped by it and the loop never ends.
# The advance is before the body for that reason, and this case is what says so:
# with the advance moved, it hangs rather than printing a wrong answer.
#
# ⛔ AND THE TARGET IS REBOUND IN ONE OF THEM. The iterator's position and the
# loop variable are two different things in CPython, and a counter loop makes
# them one unless the counter is its own name.
def show(label: str, xs: list[int]) -> None:
    print(label, xs)


out: list[int] = []
for i in range(5):
    out.append(i)
show("r5", out)

out = []
for i in range(2, 7):
    out.append(i)
show("r27", out)

out = []
for i in range(10, 0, -3):
    out.append(i)
show("rneg", out)

out = []
for i in range(0):
    out.append(i)
show("empty", out)

out = []
for i in range(5, 2):
    out.append(i)
show("backwards", out)

# continue must not skip the advance
out = []
for i in range(10):
    if i % 3 == 0:
        continue
    out.append(i)
show("cont", out)

# break, and the else that does not run
for i in range(10):
    if i == 4:
        break
else:
    print("else ran (wrong)")

for i in range(3):
    pass
else:
    print("else ran")

# rebinding the target does not move the iterator
out = []
for i in range(5):
    out.append(i)
    i = 100
show("rebind", out)

# the bounds are evaluated once
n = 3
out = []
for i in range(n):
    n = 100
    out.append(i)
show("bound-once", out)

# nested
out = []
for i in range(3):
    for j in range(2):
        out.append(i * 10 + j)
show("nested", out)

# variable step
k = 2
out = []
for i in range(0, 9, k):
    out.append(i)
show("varstep", out)

k = -2
out = []
for i in range(9, 0, k):
    out.append(i)
show("varstep-neg", out)

try:
    for i in range(0, 5, 0):
        print(i)
except ValueError as e:
    print("ValueError:", str(e))

z = 0
try:
    for i in range(0, 5, z):
        print(i)
except ValueError as e:
    print("ValueError:", str(e))

# comprehension and sum
print([i * i for i in range(6)], sum(i for i in range(10)))
# range as a value is still a range
r = range(3)
print(list(r), len(r), 2 in r)


def gen(n: int):
    for i in range(n):
        yield i * 2


print(list(gen(4)))


def guarded(n: int) -> int:
    s = 0
    for i in range(n):
        try:
            if i == 2:
                raise ValueError("x")
            s += i
        except ValueError:
            s += 100
    return s


print(guarded(5))
