# What: an unnarrowed `T | None` in an object position is refused by a
#   diagnostic that names Optional and narrowing. Rejecting it is correct
#   (pyright rejects it too, and the value's runtime form is a member tag plus
#   per-member storage, not an object handle); the defect was the wording --
#   "runtime object header has invalid type 'i64'", which reported the union's
#   TAG type and mentioned neither Optional nor what to do about it.
#   The narrowed and defaulted forms both work:
#   cases/optional_dict_get_narrowed.py is the pinned control.
d: dict[str, int] = {"a": 1, "b": 2}
print(d.get("b"))
