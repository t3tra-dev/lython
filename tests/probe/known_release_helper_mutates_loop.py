# probe: REPORTED loud (B7): a mutated list handed to a helper inside a loop
# axes: op=pass-to-function flow=for
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 16 : i32, ly.source.end_line = 7 : i32, ly.source.start_col = 4 : i32, ly.source.start_line = 7 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe
# CPython 3.14 expects: 3

def put(xs: list[int], v: int) -> None:
    xs.append(v)


xs: list[int] = []
for i in range(3):
    put(xs, i)
print(len(xs))
