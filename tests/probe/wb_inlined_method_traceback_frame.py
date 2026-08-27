# probe: a method inlined into its caller leaves no traceback frame of its own
# axes: acquire=method-call width=object op=raise flow=inline observe=traceback
# CLASSIFICATION @ 2026-08-27: 2 誤って実行する
#
# `b.bad()` is inlined by the emitter (the LLVM IR has no `@bad`; `@call` holds
# the FloorDiv), so no invoke marks the call site and no frame is pushed for it.
# The traceback then shows ONE frame where CPython shows two, and the surviving
# frame carries the CALLER's function name against the CALLEE's line -- the
# entry reads `line 6, in call` while line 6 is inside `bad`. A plain function
# call (tests/golden/cases/a_formatted_traceback_matches_cpython.py) is not
# inlined and keeps both frames.
#
# Pre-existing: `git stash && cmake --build build-rel` at 41177298 prints the
# same one-frame traceback.
#
# CPython 3.14 expects:
#   Traceback (most recent call last):
#     File "...", line 9, in <module>
#       call(Box(3))
#       ~~~~^^^^^^^^
#     File "...", line 8, in call
#       return b.bad()
#              ~~~~~^^
#     File "...", line 6, in bad
#       return self.v // 0
#              ~~~~~~~^^~~
#   ZeroDivisionError: division by zero


class Box:
    v: int

    def __init__(self, v: int) -> None:
        self.v = v

    def bad(self) -> int:
        return self.v // 0


def call(b: Box) -> int:
    return b.bad()


call(Box(3))
