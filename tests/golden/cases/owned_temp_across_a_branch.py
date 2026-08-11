# Why execution: none of these compiled or printed the right number, and the
# two defects are one.
#
#   - `t = a - b` boxes its result inside the prim/boxed dispatch's scf.if, so
#     the entity's name outside the region is the scf.if's result. The release
#     placement asked about the call INSIDE the region, where scf.yield has no
#     CFG successors, and every strategy declined or placed nothing. Any branch
#     between the arithmetic and the use did it: "owned resource from
#     @LyLong_FromI64 result 0 reaches function exit without release".
#
#   - with the placement repaired, the int global store can finally ask the
#     fast lane's VALID flag instead of storing the callee's speculative dummy,
#     which is why `COUNT = add(COUNT, i)` in a loop printed 0.


def branch_after_arithmetic(a: int, b: int) -> None:
    t = a - b
    if t == 0:
        return
    print(t)


def branch_without_early_return(a: int, b: int) -> None:
    t = a - b
    if a > 0:
        print(t)


def use_in_the_merge_block(a: int, b: int) -> None:
    t = a * b
    if a > 0:
        print("pos")
    print(t)


def returned_from_a_branch(a: int, b: int) -> int:
    t = a + b
    if a > 0:
        return t
    return t + 1


def inside_a_loop(n: int) -> None:
    for i in range(n):
        t = i * 2
        if t > 0:
            print(t)


COUNT: int = 0
TOTAL: int = 0


def add(a: int, b: int) -> int:
    return a + b


def step(i: int) -> None:
    global COUNT
    COUNT = COUNT + i


def main() -> None:
    branch_after_arithmetic(9, 4)
    branch_without_early_return(9, 4)
    use_in_the_merge_block(3, 4)
    print(returned_from_a_branch(1, 2), returned_from_a_branch(-1, 2))
    inside_a_loop(3)


main()
for i in range(4):
    TOTAL = add(TOTAL, i)
for i in range(3):
    step(i)
print(TOTAL, COUNT)
