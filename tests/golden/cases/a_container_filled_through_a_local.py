# The scan that decides an empty container's element type from what fills it
# reads the filling expression, and two ordinary ways of writing one were
# opaque to it.
#
# A local that is ITSELF an empty literal answered `list[object]` -- a type,
# so it passed the scan's test and the outer container took it, losing the
# inner element two levels down:
#
#     out = []
#     for _ in range(rows):
#         line = []
#         for _ in range(cols):
#             line.append(0)
#         out.append(line)
#     # grid(2, 2)[0][0] + 1
#     # builtins.object does not provide manifest method '__add__'
#
# And a local the NAME flowed into read the type being decided, one binding
# removed, so it disagreed with the honest seed beside it and the pair left the
# container erased -- which is every stack machine:
#
#     b = stack.pop()
#     a = stack.pop()
#     stack.append(a + b)      <- object
#     ...
#     stack.append(int(t))     <- int
#
# Why execution: the element type decides which runtime container is built and
# what may be read back out of it, so the program has to DECODE what it stored
# -- `grid[0][0] + 1` and the evaluator's arithmetic are the assertions here,
# not the fact that it compiles.
#
# ⭐ Neither parameter annotations nor their absence change the answer: the
# scan is the emitter's, and the signature walk asks the same one
# (an_inferred_function_that_builds_a_list). `grid` is annotated and
# `evaluate` is not, on purpose.
#
# ⛔ The nesting is depth-bounded, not cycle-detected: a container inside a
# container is two or three deep in real programs, and a bound is cheaper to be
# sure of than a visited set threaded through a scan that pushes type scopes.


def grid(rows: int, cols: int) -> "list[list[int]]":
    out = []
    for _ in range(rows):
        line = []
        for _ in range(cols):
            line.append(0)
        out.append(line)
    return out


def buckets(words: "list[str]") -> "dict[str, list[str]]":
    out = {}
    for w in words:
        key = w[0]
        if key not in out:
            group = []
            out[key] = group
        out[key].append(w)
    return out


def evaluate(tokens):
    stack = []
    for t in tokens:
        if t == "+":
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
        elif t == "*":
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)
        else:
            stack.append(int(t))
    return stack.pop()


g = grid(2, 3)
g[1][2] = 7
print(g)
print(g[0][0] + 1, g[1][2] * 2)

table = buckets(["ant", "arc", "bee"])
print(sorted(table.keys()))
print(table["a"][1] + "!")

print(evaluate(["2", "3", "+", "4", "*"]))
