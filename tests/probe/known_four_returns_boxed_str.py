# probe: the same shape with a str temporary instead of an int
# axes: width=w1str op=temp flow=multireturn
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 18 : i32, ly.source.end_line = 14 : i32, ly.source.start_col = 4 : i32, ly.source.start_line = 14 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: v0 v1a v2b v3c

def pick(k: int) -> str:
    t = "v" + str(k)
    if k == 0:
        return t
    if k == 1:
        return t + "a"
    if k == 2:
        return t + "b"
    return t + "c"


print(pick(0), pick(1), pick(2), pick(3))
