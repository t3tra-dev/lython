# The same re-raise-inside-a-loop shape as try_reraise_loop_carried_str, with
# TWO owned str locals rebound by the outer handler and carried across the loop
# header together.
#
# What the second local adds. The unwind cleanup for one iteration has to
# release the incoming generation of each carried group exactly once, and the
# chain that discharges it is built per requirement (handler, group set): with
# one group a planner that computes the set at the shared handler instead of at
# the incoming edge still gets a plausible answer, with two it has to get the
# same answer twice from the same question. `i` (int) is a third carried group
# with a different deallocator, so the releaser's argument layout is exercised
# as well.
#
# `seen.append(i)` was the first spelling tried here and is deliberately absent:
# a list mutated inside a try inside a loop is refused today for an unrelated
# reason (list still declares transfer_args, the loop-carried transferred-
# receiver shape in rfc/lane-conversion-playbook.md), so it would have pinned
# that defect instead of this one.
def h(n: int) -> str:
    tag = "start"
    last = "none"
    i = 0
    while i < n:
        try:
            try:
                raise ValueError("inner")
            except ValueError:
                raise KeyError("outer")
        except KeyError as e:
            tag = "hit"
            last = str(e)
        i += 1
    return tag + "/" + last


print(h(0))
print(h(3))
print(h(200))
