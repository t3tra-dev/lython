# probe: REPORTED loud: dict setitem on a borrowed (parameter) dict
# axes: acquire=param width=w1dict op=setitem flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 10 : i32, ly.source.end_line = 7 : i32, ly.source.start_col = 4 : i32, ly.source.start_line = 7 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe
# CPython 3.14 expects: 2 2

def put(d: dict[str, int]) -> None:
    d["b"] = 2


d: dict[str, int] = {"a": 1}
put(d)
print(len(d), d["b"])
