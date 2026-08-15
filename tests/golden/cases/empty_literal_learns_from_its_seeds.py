# What this pins: `xs = []`, `d = {}` with no annotation learn their element
# types from the operations that seed them. An empty literal has nothing of its
# own to infer from, so it typed as `list[object]` and the next line stopped
# compiling -- `xs.append(1); xs[0] + 1` was "'builtins.object' does not
# provide manifest method '__add__'", which is the most common way a Python
# program builds a list, a dict or a set.
#
# Why this needs to run rather than assert on a diagnostic: the seed decides
# what the container HOLDS, so getting it wrong is a different program, not a
# refusal. A dict seeded `d[w] = 1` under a `for w in words` has to reach
# `str -> int`, and reading the key type off the wrong place would compile and
# index with the wrong thing. The counts below are what separate them.
#
# ⛔ Two boundaries are deliberately exercised as the control, because the
# scan must DECLINE rather than guess:
#   - seeds that disagree (`xs.append(1); xs.append("s")`) leave the literal at
#     object, which is where it was and which CPython allows;
#   - a seed whose own type is not known yet -- an empty literal appended into
#     another empty literal -- is out of reach and stays refused
#     (`tests/probe/wb_grid_leftovers_2026_08_16.py`).
#
# Every expected line is python3.14's.

# --- the straight-line shapes ---------------------------------------------
xs = []
xs.append(1)
print(xs, xs[0] + 1, len(xs))

names = []
names.append("a")
names.append("b")
print(names, names[0].upper(), ",".join(names))

table = {}
table["a"] = 1
print(sorted(table.items()), table["a"] + 1)


# --- seeded from a loop variable ------------------------------------------
squares = []
for i in range(4):
    squares.append(i * i)
print(squares, sum(squares), squares[3] + 1)

widths = {}
for word in ["aa", "b", "ccc"]:
    widths[word] = len(word)
print(sorted(widths.items()), widths["b"] + 1)


# --- the frequency count, whose own seed reads the dict --------------------
# `counts[w] = counts[w] + 1` mentions `counts`, so it is skipped rather than
# counted as disagreement; `counts[w] = 1` beside it is what decides.
counts = {}
for w in ["a", "b", "a", "c", "a"]:
    if w in counts:
        counts[w] = counts[w] + 1
    else:
        counts[w] = 1
print(sorted(counts.items()), counts["a"] * 2)


# --- inside a function, and two literals in one suite ----------------------
def collect(n: int) -> int:
    out = []
    k = 0
    while k < n:
        out.append(k + 1)
        k += 1
    return sum(out)


print(collect(4))

lefts = []
rights = []
for v in [1, 2, 3, 4]:
    if v % 2 == 0:
        lefts.append(v)
    else:
        rights.append(v)
print(lefts, rights, lefts[0] + rights[0])


# --- the control: disagreeing seeds stay as they were ---------------------
mixed = []
mixed.append(1)
mixed.append("s")
print(len(mixed))


# --- and an annotation still wins -----------------------------------------
declared: list[int] = []
declared.append(7)
print(declared, declared[0] + 1)
