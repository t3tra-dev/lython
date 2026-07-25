# probe: borrowed set add (grow)
# axes: acquire=param width=w1set op=add flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 12 : i32, ly.source.end_line = 7 : i32, ly.source.start_col = 4 : i32, ly.source.start_line = 7 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe
# CPython 3.14 expects: 2

def put(s: set[int]) -> None:
    s.add(2)


s: set[int] = {1}
put(s)
print(len(s))
