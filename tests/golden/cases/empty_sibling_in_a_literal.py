# What this pins: an EMPTY container literal sitting beside a non-empty one
# takes the sibling's element type. It has none of its own, so it typed as
# `list[object]`, and the join gave `list[int] | list[object]` -- a union of two
# list types that nothing accepts:
#
#     g = {0: [1, 2], 1: []}
#     len(g[0])
#     # '!py.union<list[int], list[object]>' does not provide '__len__'
#
# which is every adjacency map, bucket table and grouping with one empty entry.
# An empty literal beside a typed one is not evidence of heterogeneity, it is
# an absence of evidence.
#
# Why this needs to run rather than assert on a diagnostic: the repair is in
# TWO places that have to agree. The join decides what the container SAYS it
# holds, and the element emission decides what the recorded evidence says --
# fixing only the join left `buckets["b"].append(2)` refused with "evidence
# contract 'list[object]' is not assignable to result 'list[int]'". Only
# mutating through the empty entry and reading it back shows they agree.
#
# ⛔ The control is `[[], []]`: nothing there has a type to take, so it keeps
# the erased element it always had, and `[1, "a"]` keeps its join. A repair
# that handed every element the container's joined type would retype the ones
# that decided it.
#
# Every expected line is python3.14's.


# --- a dict whose values are lists, one of them empty ---------------------
graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
print(len(graph[0]), len(graph[3]), sorted(graph.keys()))
graph[3].append(0)
print(graph[3], len(graph[3]))


def neighbours(g: dict[int, list[int]], node: int) -> int:
    return len(g[node])


print(neighbours(graph, 0), neighbours(graph, 3))


# --- a list of lists with an empty one in the middle ----------------------
rows = [[1, 2], [], [3]]
print(rows, len(rows[0]), len(rows[1]), sum(rows[2]))
rows[1].append(9)
print(rows, sum(rows[1]))


# --- a bucket table built from a literal and then filled ------------------
buckets = {"a": [1], "b": []}
buckets["b"].append(2)
buckets["a"].append(3)
print(sorted(buckets.items()))


# --- a dict of dicts, one empty -------------------------------------------
nested = {"x": {"a": 1}, "y": {}}
print(sorted(nested.keys()), nested["x"]["a"], len(nested["y"]))
nested["y"]["b"] = 2
print(sorted(nested["y"].items()))


# --- the controls ---------------------------------------------------------
# Nothing to take a type from, so the erased element stays.
allempty = [[], []]
print(allempty, len(allempty))
# A real join is still a join.
mixed = [1, "a"]
print(mixed)
# And the elements that DECIDED the join keep their own types.
widened = [1, 2.5]
print(widened)
