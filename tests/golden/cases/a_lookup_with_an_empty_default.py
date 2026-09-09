# An empty container literal has no element type of its own, and a sibling
# already knew not to count one -- `{0: [1, 2], 1: []}` takes its element from
# the full entry. The join over an unannotated function's RETURNS did not:
#
#     def get(table, key):
#         if key not in table:
#             return []
#         return table[key]
#     print(get({"a": [1, 2]}, "a")[0] + 1)
#     # builtins.object does not provide manifest method '__add__'
#
# `list[object]` joined with `list[int]` is a union of two list types that
# nothing accepts, and that is the shape of every lookup with a default. The
# annotated spelling always worked, because there the annotation IS the element
# type.
#
# Why execution: the result type decides what the caller may read out of what
# it got back, and both arms have to still produce what CPython produces --
# the empty one included.
#
# ⭐ The same rule, from the same place: `joinIgnoringEmptyLiterals` drops the
# empty entries when anything else contributed, and keeps them when nothing
# did, so a function whose every return is empty still answers with a container
# rather than nothing.
#
# ⛔ Only an empty CONTAINER. `return 0` beside `return table[key]` is a real
# type on both sides and the join stands -- an int and a str really are a
# union, and the refusal that follows is about the program.


def get(table, key):
    if key not in table:
        return []
    return table[key]


def names(rows, want):
    if want == "":
        return []
    out = []
    for r in rows:
        if r[0] == want:
            out.append(r[1])
    return out


def always_empty(flag):
    if flag:
        return []
    return []


table = {"a": [1, 2], "b": [3]}
print(get(table, "a")[0] + 1)
print(get(table, "b"), get(table, "zz"))

rows = [("x", "ann"), ("y", "bob"), ("x", "cid")]
print(names(rows, "x"))
print(names(rows, "x")[1] + "!")
print(names(rows, ""))

print(always_empty(True), always_empty(False))
