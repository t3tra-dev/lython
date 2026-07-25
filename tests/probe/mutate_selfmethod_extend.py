# probe: in-place mutation through self inside a method -- list field extend
# axes: acquire=self width=w3list/w1dict op=extend flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 29 : i32, ly.source.end_line = 11 : i32, ly.source.start_col = 8 : i32, ly.source.start_line = 11 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 3 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v

    def touch(self) -> None:
        self.f.extend([2, 3])

    def show(self) -> None:
        print(len(self.f), self.f[2])


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


o = mk()
o.touch()
o.show()
