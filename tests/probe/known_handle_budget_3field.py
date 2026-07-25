# probe: REPORTED loud (budget 5): a 3-field class used as a list element
# axes: width=wNcls op=store-into-container flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 34 : i32, ly.source.end_line = 13 : i32, ly.source.start_col = 18 : i32, ly.source.start_line = 13 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pr
# CPython 3.14 expects: 1 2 3

class Three:
    def __init__(self, a: int, b: int, c: int) -> None:
        self.a: int = a
        self.b: int = b
        self.c: int = c


xs: list[Three] = [Three(1, 2, 3)]
print(xs[0].a, xs[0].b, xs[0].c)
