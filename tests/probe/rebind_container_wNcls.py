# probe: field rebind -- receiver read out of a list; field type Wide
# axes: acquire=container width=wNcls op=rebind flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 25 : i32, ly.source.end_line = 28 : i32, ly.source.start_col = 19 : i32, ly.source.start_line = 28 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pr
# CPython 3.14 expects: 1 2.5 z

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Wide:
    def __init__(self, a: int, b: float, c: str) -> None:
        self.a: int = a
        self.b: float = b
        self.c: str = c


class Box:
    def __init__(self, v: Wide) -> None:
        self.f: Wide = v


def mk() -> Box:
    v: Wide = Wide(0, 0.0, "")
    return Box(v)


boxes: list[Box] = [mk()]
o = boxes[0]
fresh: Wide = Wide(1, 2.5, "z")
o.f = fresh
print(o.f.a, o.f.b, o.f.c)
