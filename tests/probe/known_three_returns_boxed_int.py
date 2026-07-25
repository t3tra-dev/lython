# probe: REPORTED uncompilable: a boxed int temporary live across three returns
# axes: width=w3int op=temp flow=multireturn
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 16 : i32, ly.source.end_line = 14 : i32, ly.source.start_col = 4 : i32, ly.source.start_line = 14 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 7 1000008 2000009 3000010

def pick(k: int) -> int:
    t = k * 1000000 + 7
    if k == 0:
        return t
    if k == 1:
        return t + 1
    if k == 2:
        return t + 2
    return t + 3


print(pick(0), pick(1), pick(2), pick(3))
