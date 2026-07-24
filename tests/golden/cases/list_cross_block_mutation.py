# A list mutated inside branch arms and loop bodies reads back its real
# contents after the merge: the evidence tier is demoted at a cross-block
# mutation, so post-merge reads go through the shared physical payload
# instead of one predecessor's non-dominating element evidence.
xs: list[int] = [1, 2, 3]
n = 4
if n > 2:
    xs.append(n)
else:
    xs.append(-n)
if n > 3:
    xs.append(n * 10)
for v in xs:
    print(v)
print(xs)
print(len(xs))
print(xs[3])
print(xs[-1])


def branch_arms(k: int) -> None:
    ys: list[int] = [1, 2, 3]
    if k > 2:
        ys.append(k)
    else:
        ys.append(-k)
    print(ys)
    print(len(ys))
    print(ys[3])


branch_arms(4)
branch_arms(1)


def loop_body(count: int) -> None:
    zs: list[int] = [0]
    for i in range(count):
        zs.append(i * i)
    print(zs)
    print(len(zs))


loop_body(4)


def conditional_in_loop(count: int) -> None:
    picked: list[int] = []
    for i in range(count):
        if i % 2 == 0:
            picked.append(i)
        else:
            picked.append(-i)
    print(picked)
    print(len(picked))


conditional_in_loop(5)


def store_in_branch(k: int) -> None:
    ws: list[int] = [1, 2, 3]
    if k > 2:
        ws[0] = 9
        ws[2] = k
    print(ws)


store_in_branch(4)
store_in_branch(1)
