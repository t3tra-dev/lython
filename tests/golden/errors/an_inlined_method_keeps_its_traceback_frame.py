# WHAT: the traceback of an exception raised inside a method, printed by the
# runtime's own uncaught-exception printer -- three frames, each with its own
# name, line and `~~~^^^` anchors, exactly as CPython 3.14 prints them.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: a method body is written
# INTO its caller, so at every layer below this one there is no frame to check.
# The frames exist only in what the running program prints, and the failure this
# guards was a traceback that still read plausibly: one frame instead of three,
# with the innermost line under the outermost name.
#
# ⛔ THE LAST LINE IS A CHAINED CALL on purpose. Its anchors are the ones the
# "split at the first `(`" heuristic got wrong -- it underlined the arguments of
# `Box` instead of the call being made -- and only a chain shows the difference.


class Box:
    v: int

    def __init__(self, v: int) -> None:
        self.v = v

    def inner(self) -> int:
        return self.v // 0

    def bad(self) -> int:
        return self.inner()


Box(3).bad()
