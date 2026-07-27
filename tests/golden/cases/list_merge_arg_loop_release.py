# Why this needs execution: the dropped borrow-edge retain is invisible until the
# loop's release actually RUNS -- each list group's own retain/release arithmetic
# balances, so the affine verifier reports nothing and the program compiles clean
# under --release. On main before this case it aborted with `Ly_DecRef observed
# non-positive refcount` (exit 134).
#
# The shape: `ys` is a merge argument reconciling TWO one-lane `builtins.list`
# groups, one of which is `xs` itself, so the retain that lends the merge argument
# its token has to be written through a prefix view of a memref<9xi64> handle that
# is a call result rather than a block argument.


def run(n: int) -> int:
    total = 0
    for i in range(n):
        xs: list[int] = [i]
        ys: list[int] = xs if i % 2 == 0 else [i, i]
        total += len(ys)
        total += len(xs)
    return total


print(run(4))
print(run(1))
