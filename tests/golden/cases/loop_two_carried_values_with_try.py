# Why this needs execution: it pins the VALUE two loop-carried block arguments
# accumulate when expanding one of them re-enters the expansion of the other. The
# arity failure it was found through (`branch has N operands for successor #0, but
# target block has N+1`) is caught before execution, but the repair is an ORDER of
# edge splicing, and only running the loop shows that both accumulators still
# forward the right values on every edge rather than merely a well-formed count.
#
# Two directions, because the repair is direction-sensitive and got that backwards
# once: `accumulate_list` nests a LATER block argument inside an earlier one (the
# int accumulator's back edge reads len(xs)), `accumulate_str` nests an EARLIER one
# inside a later one. The second compiled before the repair and must keep working.


def accumulate_list(n: int) -> int:
    total = 0
    i = 0
    xs: list[int] = [0]
    while i < n:
        try:
            total += 1
        except ValueError:
            total += 2
        xs = xs + [i]
        total += len(xs)
        i += 1
    return total


def accumulate_str(n: int) -> int:
    total = 0
    i = 0
    s: str = "a"
    while i < n:
        try:
            total += 1
        except ValueError:
            total += 2
        s = s + "b"
        total += len(s)
        i += 1
    return total


print(accumulate_list(3))
print(accumulate_list(0))
print(accumulate_str(3))
