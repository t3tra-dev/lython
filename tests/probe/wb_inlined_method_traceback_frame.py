# probe: a method inlined into its caller keeps a traceback frame of its own
# axes: acquire=method-call width=object op=raise flow=inline observe=traceback
# CLASSIFICATION @ 2026-08-27: 1 正しい
#
# `b.bad()` is inlined by the emitter -- the LLVM IR has no `@bad`; `@call`
# holds the FloorDiv -- so no invoke marks the call site and nothing named the
# body. The traceback showed ONE frame where CPython shows three, and the
# surviving one carried the CALLER's name against the CALLEE's line: `line 6,
# in call` where line 6 is inside `bad`.
#
# The frames an inlined body would have contributed now ride in the location
# (`ly.source.function` / `ly.source.inline_at`, written by the emitter) and are
# pushed one per level, innermost first, by both frame-pushing paths: the
# invoke cleanup for a raise that happens in a callee, and the raise-site push
# for one the lowering inlined too (`b.at(9)`'s bounds check raises in the
# caller, not in a callee).
#
# CPython 3.14 expects:
#   Traceback (most recent call last):
#     File "...", line 44, in <module>
#       call(Box(3))
#       ~~~~^^^^^^^^
#     File "...", line 41, in call
#       return b.bad()
#              ~~~~~^^
#     File "...", line 37, in bad
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
