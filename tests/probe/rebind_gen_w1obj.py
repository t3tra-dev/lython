# probe: field rebind -- receiver from a generator yield; field type Other
# axes: acquire=gen width=w1obj op=rebind flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 14 : i32, ly.source.end_line = 35 : i32, ly.source.start_col = 9 : i32, ly.source.start_line = 35 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 7

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Wide:
    def __init__(self, a: int, b: float, c: str) -> None:
        self.a: int = a
        self.b: float = b
        self.c: str = c


class Box:
    def __init__(self, v: Other) -> None:
        self.f: Other = v


def mk() -> Box:
    v: Other = Other(0)
    return Box(v)


from typing import Iterator


def gen() -> Iterator[Box]:
    yield mk()


for o in gen():
    fresh: Other = Other(7)
    o.f = fresh
    print(o.f.n)
