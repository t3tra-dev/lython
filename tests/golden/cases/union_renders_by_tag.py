# Why execution: an unnarrowed `T | None` could not be rendered at all --
# "unnarrowed !py.union<...> cannot be used where a concrete object is
# required" from the lowering, or "print() cannot render argument N" from the
# emitter. A union has no class, no manifest contract and no header of its own
# to dispatch on, so it renders by TESTING ITS TAG and rendering the member
# that is live. Only running shows the right member was picked: a chain that
# tests in the wrong order, or renders an inactive member, reads a zeroed
# placeholder rather than failing.
#
# This replaces errors/optional_unnarrowed_object_position, which pinned the
# refusal. CPython prints the value; there was never anything to refuse.
d: dict[str, int] = {"a": 1, "b": 2}
print(d.get("b"))
print(d.get("zz"))
print(d.get("a"), d.get("zz"))


def pick_int(flag: bool) -> int | None:
    if flag:
        return 7
    return None


def pick_str(flag: bool) -> str | None:
    if flag:
        return "hi"
    return None


# Single argument and multiple, which take different paths into the renderer.
print(pick_int(True))
print(pick_int(False))
print(pick_int(True), pick_int(False))
print(pick_str(True), pick_str(False))

# The other spellings of the same ladder.
print(str(pick_int(True)), str(pick_int(False)))
print(f"{pick_int(True)}/{pick_str(False)}")
print("v=" + str(pick_str(True)))

# ⛔ A three-member union with TWO owning members (`int | str | None`) is not
# here: the renderer handles it, but returning one is refused for an unrelated
# reason -- "conditionally owned resource ... reaches function exit without
# tag-conditioned release", the conditional-group hole recorded at the
# `g.condition` skip in Passes/Ownership.cpp
# (tests/probe/wb_three_member_union_return.py).

# Narrowing still wins where it applies -- the renderer is for what is left.
v = pick_int(True)
if v is not None:
    print(v + 1)
print(v)
