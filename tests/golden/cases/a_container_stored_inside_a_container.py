# The scan that decides an empty container's element type reads the expression
# that fills it, and an empty container stored INTO one was read rather than
# asked: it answers `list[builtins.object]` on its own, which is a type, so the
# outer container took it and the inner element was lost.
#
#     out = {}
#     for w in words:
#         if w[0] not in out:
#             out[w[0]] = []
#         out[w[0]].append(w)      <- nothing looked here
#     # out["a"][1] + "!"
#     # builtins.object does not provide manifest method '__add__'
#
# which is the adjacency map, the grouping and the bucket table -- and the same
# shape written through a LOCAL (`bucket = []; out[k] = bucket`) already
# compiled, one spelling over.
#
# Why execution: the inner element type decides what may be read back out, so
# the program has to DECODE what it stored -- the concatenations and the
# arithmetic below are the assertions, not the fact that it compiles.
#
# ⭐ ONE SCAN, ASKED ONE SUBSCRIPT DEEPER. `name[...]` is how every operation
# on a container inside another one is spelled, so the scan takes the depth as
# a parameter and its existing arms answer unchanged. Dict-of-list,
# list-of-list, dict-of-set and dict-of-dict all fall out of that.
#
# ⛔ `bucket = out.setdefault(k, [])` is still refused: the local there takes
# the CALL'S RESULT, which is not one of the shapes this scan reads, and the
# empty literal it is given is an argument rather than a store.
#
# ⛔ Depth-bounded, and it falls back to reading the empty literal when the
# deeper scan finds nothing -- the erased container is what this answered
# before, and a program that only prints the outer one never decodes an inner
# element.


def group(words):
    out = {}
    for w in words:
        k = w[0]
        if k not in out:
            out[k] = []
        out[k].append(w)
    return out


def rows(n: int):
    out = []
    for i in range(n):
        out.append([])
        out[i].append(i * 10)
    return out


def index(words):
    out = {}
    for w in words:
        out[w] = {}
        out[w]["len"] = len(w)
    return out


def unique(words):
    out = {}
    for w in words:
        k = w[0]
        if k not in out:
            out[k] = set()
        out[k].add(w)
    return out


names = ["ant", "arc", "bee", "ant"]
buckets = group(names)
print(sorted(buckets.keys()))
print(buckets["a"][1] + "!")
print(len(buckets["a"]), len(buckets["b"]))

grid = rows(3)
print(grid)
print(grid[2][0] + 1)

sized = index(["ab", "cde"])
print(sized["cde"]["len"] + 1)

seen = unique(names)
print(sorted(seen["a"]), len(seen["a"]))
