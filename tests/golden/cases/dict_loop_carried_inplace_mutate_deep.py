# The same loop-carried in-place dict mutation as
# dict_loop_carried_inplace_mutate, run enough times that the native stack frame
# would have to grow to survive it.
#
# Why this needs to execute rather than assert on IR: the defect it pins is not a
# wrong value or a rejected program but the SIZE OF ONE STACK FRAME, and a frame
# that grows per iteration is only observable by iterating.
# DriverTest.BoxedContainerLoopKeepsPayloadSlotsOutOfTheLoopBody asserts the
# structural half (no payload-box slot inside a loop body); this asserts the
# consequence, which also covers slots this lowering does not own -- the
# descriptor spill the memref->LLVM conversion used to emit beside a size query
# was a second, smaller instance of the same growth and no box-shaped assertion
# would have seen it.
#
# The trip counts are the measurement, not a guess. Before the fix, the boxed
# setitem left two 16-word `memref.alloca`s in the loop body and the frame grew
# ~256 bytes per iteration: the re-store loop below stopped working between
# 25,000 and 30,000 iterations on an 8 MB stack (verified on the pre-fix binary).
# 1,000,000 iterations demand ~256 MB, which is past every stack limit this
# project can be configured with, so the case cannot go quietly green on a host
# with a generous `ulimit -s`. After the fix both loops run in a constant frame;
# 10,000,000 iterations of the second one also pass.
#
# Mutating an EXISTING key keeps the size stable, so the iteration guard does not
# raise and both loops run to completion -- the path that pays the stack.
d: dict[int, int] = {}
for i in range(4):
    d[i] = i

# 4,000,000 executions of the iterate-and-mutate body.
acc = 0
for j in range(1000000):
    for k in d:
        d[0] = k
        acc = acc + 1
print(acc)

# 1,000,000 executions of the plain re-store body, which boxes a key and a value
# per iteration without an active dict iterator.
n = 0
for j2 in range(1000000):
    d[0] = j2
    n = n + 1
print(n, d[0])
print(len(d), d[1], d[3])
