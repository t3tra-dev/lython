# A BARE `raise` inside a handler, inside a loop that completes.
#
# Why this is a separate case from try_reraise_loop_carried_str: a bare re-raise
# resumes the exception already in flight rather than constructing one, so the
# handler holds no new exception token and the unwind edge out of the inner
# handler carries only the loop-carried locals. That is the same planner path
# with one fewer group at the exit point, and the two spellings answer
# differently if the planner's held-token question is decided by what the
# handler block constructs instead of by what the incoming unwind holds.
def k(n: int) -> str:
    out = "none"
    i = 0
    while i < n:
        try:
            try:
                raise ValueError("boom")
            except ValueError:
                raise
        except ValueError as e:
            out = str(e)
        i += 1
    return out


print(k(0))
print(k(2))
print(k(200))
