# probe: REPORTED loud: list append on a borrowed (parameter) list
# axes: acquire=param width=w3list op=append flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 16 : i32, ly.source.end_line = 7 : i32, ly.source.start_col = 4 : i32, ly.source.start_line = 7 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe
# CPython 3.14 expects: 2 2

def put(xs: list[int]) -> None:
    xs.append(2)


xs: list[int] = [1]
put(xs)
print(len(xs), xs[1])
