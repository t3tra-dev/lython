# A handler that RAISES, inside a loop that COMPLETES, with a str local carried
# across the loop header and rebound by the outer handler.
#
# What this pins, and why the rest of the try/except suite did not. The outer
# handler's rebind makes `out` a loop-carried owner group, so the loop latch
# releases the incoming generation and forwards the new one. The inner handler's
# raise puts a second unwind edge inside the same iteration, which is what
# forces the unwind-cleanup planner to run twice over the same block-argument
# group -- once for the try's own handler and once for the may-raise calls after
# it (`i += 1` lowers to LyLong_FromI64 + LyLong_Add). Every other exception
# golden here either leaves the loop through the exception (exception_loop_
# carried_rebuild, dict_changed_size) or has no re-raise (except_handler_rebind_
# carry), so the completing path with two nested unwind edges was never
# compiled.
#
# g(0) is not redundant: it keeps the never-entered-loop path live, so the
# initial "none" generation has to be returned without a release, and it is the
# only spelling in this file where the header argument is the producer's value
# rather than a previous iteration's.
def g(n: int) -> str:
    out = "none"
    i = 0
    while i < n:
        try:
            try:
                raise ValueError("inner")
            except ValueError:
                raise KeyError("outer")
        except KeyError as e:
            out = str(e)
        i += 1
    return out


print(g(0))
print(g(2))
# Enough iterations that a per-iteration refcount imbalance on the NORMAL path
# would be observable rather than incidental.
print(g(200))
