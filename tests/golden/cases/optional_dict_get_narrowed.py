# What: the working counterparts of errors/optional_unnarrowed_object_position
#   -- `dict.get(k)` narrowed with `is not None`, and the two-argument form
#   whose result is not Optional at all. Pinned as a pair with that error case
#   so the refusal there stays a statement about NARROWING and not about
#   dict.get or about Optional being unrepresentable.
# This one is a CONTROL: it was green before the diagnostic change too
# (redcheck reports it as never-red, by design). Its job is to make the error
# case's claim falsifiable, not to catch a regression in that change.
d: dict[str, int] = {"a": 1, "b": 2}

v: int | None = d.get("b")
if v is not None:
    print(v)
else:
    print("missing")

w: int | None = d.get("zz")
if w is not None:
    print(w)
else:
    print("missing")

print(d.get("zz", 0))
print(d.get("b", 0))
