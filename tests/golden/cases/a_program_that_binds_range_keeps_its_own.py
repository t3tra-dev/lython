# WHAT: a module that defines `range` itself gets its own, in loop position too.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the wrong answer here is a
# RUNNING program with different output -- the loop counts 0,1,2,... instead of
# walking what the program returned -- which is the one shape a refusal can
# never catch.
#
# ⛔ The guard is `programBindsName`, not the iterator-name test the other
# fusions use: that one also refuses a name the type system knows as a class,
# and `range` is bound there as `builtins.range`, so it is false for `range`
# always and the rewrite never fires at all.


def range(n: int) -> list[int]:
    return [n, n + 1]


out: list[int] = []
for v in range(7):
    out.append(v)
print(out)

total = 0
for v in range(20):
    total += v
print(total)
