# What this pins: `.get()` on a dict whose values are a UNION.
#
#     doc = {"id": 1, "name": "x"}
#     print(doc.get("id"))
#     # type mismatch for bb argument #0 of successor #0
#
# `.get` merges two arms into `V | None`: the absent arm's None was wrapped
# into the union and the present arm's value was not, because the coercion asks
# whether the value's type IS one member (a union never is) and then whether it
# is assignable TO one (it is not either). So the value went to the merge
# unwrapped and the block argument disagreed with it. A NARROWER union injects
# into a wider one, which is what `py.union.wrap` already does -- it remaps the
# tag member by member -- so this is the emitter asking for what the lowering
# can already perform.
#
# It only shows for a heterogeneous dict: `{"id": 1}.get("id")` is `int | None`
# and the present arm is a plain int, which the member loop handles.
#
# Why this needs to run rather than assert on a diagnostic: the injection is a
# TAG REMAP -- member i of the source becomes member j of the target -- and the
# wrong permutation compiles. `doc.get("id")` printing "x" instead of 1 is what
# a remap that dropped the source tag looks like.
#
# ⛔ Four `.get` shapes remain refused, and none of them is this defect, which
# is why each key below is read exactly once. The SAME key read twice in one
# function is "owned resource ... reaches function exit without release"; a
# FLOAT value is refused for the merge edge's retain, with no union in sight
# (`{"s": 2.5}.get("s")`); a COMPUTED key needs the dynamic evidence arm to
# select between candidates of different physical shapes; a literal key that is
# ABSENT is a read the evidence cannot answer. All four are recorded in
# tests/probe/wb_grid_leftovers_2026_08_16.py, the float one with the
# measurement of the repair that was built for it and reverted.
#
# Every expected line is python3.14's.

# --- a record read through .get, each key once ----------------------------
doc = {"id": 1, "name": "x", "tag": "t", "n2": 7, "flag": True}
print(doc.get("id"))
print(doc.get("name"))
print(doc.get("n2"))

v = doc.get("tag")
if isinstance(v, str):
    print(v.upper())
if doc.get("flag") is None:
    print("missing")
else:
    print("present")


# --- the same through an annotated dict -----------------------------------
table: dict[str, int | str] = {"a": 1, "b": "two"}
print(table.get("a"), table.get("b"))


# --- THE CONTROL: a homogeneous dict, which always worked ------------------
counts: dict[str, int] = {"a": 1}
print(counts.get("a"), counts.get("zz"))
plain = {"only": 1}
print(plain.get("only"))
