# What: exceptions unwinding THROUGH frames without a local try still leave
# every frame-owned value released (Wave 0 hand-off blind spot: the may-raise
# call in a try-less intermediate frame had no unwind cleanup; the affine
# verifier now rejects the shape unless the release-then-rethrow handlers are
# wired, so this compiling and printing correctly pins the closure).


def raiser(flag: int) -> str:
    tag = "tag-" + str(flag)
    if flag > 0:
        raise ValueError("boom " + tag)
    return tag


def middle(flag: int) -> str:
    held = "held-" + str(flag)
    inner = raiser(flag)
    return held + ":" + inner


def middle_loop(n: int) -> str:
    acc = "acc" + ""
    for i in range(n):
        acc = acc + "." + raiser(0)
        if i == 2:
            raiser(1)
    return acc


def top(flag: int) -> None:
    try:
        print(middle(flag))
    except ValueError as error:
        print("caught " + str(error))


def top_loop(n: int) -> None:
    try:
        print(middle_loop(n))
    except ValueError as error:
        print("loop caught " + str(error))


def rethrow_through_finally(flag: int) -> None:
    keep = "keep-" + str(flag)
    try:
        try:
            raiser(flag)
        finally:
            print("finally " + keep)
    except ValueError:
        print("recovered " + keep)


top(0)
top(1)
top_loop(2)
top_loop(4)
rethrow_through_finally(0)
rethrow_through_finally(1)
print("done")
