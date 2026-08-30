# `if value is not None:` on a THREE-member union left the value un-narrowed:
# the guard proves `int | str`, and the call below it was still handed
# `int | str | None`. Two members narrow to one and worked; three narrow to a
# SUB-UNION, which is not a member of the source union, so the narrowing
# silently did nothing and the call was refused.
#
# Golden because the repair emits a real `union.unwrap` that REMAPS THE TAG
# member by member: a wrong remap reads the value as the wrong member, which
# prints a plausible answer rather than failing.
def describe(value: "int | str") -> str:
    if isinstance(value, str):
        return "str:" + value
    return "int:" + str(value)


def outer(value: "int | str | None") -> str:
    if value is not None:
        return describe(value)
    return "none"


print(outer(3), outer("abc"), outer(None))


def sized(value: "list[int] | str | None") -> int:
    if value is None:
        return -1
    if isinstance(value, str):
        return len(value)
    return len(value)


print(sized([1, 2, 3]), sized("ab"), sized(None))


# ⛔ THE NARROWED UNION IS STILL A UNION, and what you may do with one has not
# changed: subscripting or iterating a narrowed `list[int] | tuple[int, int]`
# is still refused, with or without an explicit else. That is a different gap
# and this case does not reach for it -- both functions above narrow and then
# DISPATCH, which is what the guard is for.
