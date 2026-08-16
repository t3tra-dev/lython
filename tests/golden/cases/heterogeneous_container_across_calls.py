# What this pins: a heterogeneous container that is ALSO iterated, and one
# whose element is passed to a function.
#
# Two demotions of the contents evidence were wider than what they protect
# against, and for a union-element container a demotion is not a slower read
# but a refusal -- the runtime tier has no `__getitem__` that returns a union.
#
#   `cfg.keys()` is a `py.iter`, which was not on the read-only list, so
#   reading a field after listing the keys was "runtime manifest has no
#   builtins.dict.__getitem__ method". An iterator is handed out instead of the
#   container, and there is no path from it back to a mutation of this value.
#
#   `render(vals[0])` demoted the CONTAINER the argument was read out of,
#   because a callee can mutate a mutable element in place and the element map
#   would then be stale. A callee cannot mutate an int or a str. The demotion
#   is kept for an element that can be changed, including a union with a
#   mutable member -- the tag decides which, and either way the callee holds
#   something it can change.
#
# Why this needs to run rather than assert on a diagnostic: what each read
# returns is the question. An element map that survived a call it should not
# have prints the value the callee replaced, and nothing but the printed value
# says which description answered. The mutating controls at the end are the
# other half: their reads must come from the object.
#
# Every expected line is python3.14's.

# --- a config record, read after its keys are listed -----------------------
cfg = {"host": "localhost", "port": 8080, "debug": True}
print(sorted(cfg.keys()))
host = cfg["host"]
port = cfg["port"]
if isinstance(host, str) and isinstance(port, int):
    print(f"{host}:{port}")
if cfg["debug"]:
    print("debug")
print(len(cfg))


# --- iterated with `for`, then read ---------------------------------------
row = {"a": 1, "b": "x"}
for k in sorted(row.keys()):
    print(k)
print(row["a"], row["b"])


# --- an element passed to a function, then a later read --------------------
def render(v: int | str) -> str:
    if isinstance(v, int):
        return "#" + str(v)
    return v


vals: list[int | str] = [1, "b", 3]
print(render(vals[0]), render(vals[1]), render(vals[2]))
print(vals[0], vals[2])

plain = [1, "b"]
print(render(plain[0]))
print(plain[1])


# --- THE CONTROL: a MUTABLE element passed to a function still demotes -----
# The callee grows the element in place, so the outer map's description of it
# is stale the moment the call returns and the read has to come from the
# object.
def grow(v: list[int]) -> None:
    v.append(2)


data: list[list[int]] = [[1]]
grow(data[0])
print(data[0], len(data[0]), data[0][1])

nested: list[list[int]] = [[1], [5]]
grow(nested[1])
print(nested[0], nested[1], len(nested[1]))
