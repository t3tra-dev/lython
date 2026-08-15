# What this pins: `x.f is None` and a narrowed read of an Optional FIELD whose
# member has more than one runtime lane. `str | None`, `list[int] | None` and
# `bytes | None` all reach the same code as `int | None`, which has one lane and
# happened to survive it.
#
# Why this needs to run rather than assert on a diagnostic: the repair is not
# the refusal, it is WHICH VALUE the tag comes from. The read used to hand back
# the cached member bundle -- the str -- so `union.test` compared the str's
# header memref against the tag constant. That was caught by an ABI check
# ("runtime bundle value 0 for 'builtins.bool' has type 'memref<2xi1>'"), and a
# one-lane member would have compared a header address against 0 or 1 and
# answered silently. Only the printed True/False separates a tag read from the
# right lane from one read off the wrong lane.
#
# ⛔ Four neighbouring shapes are deliberately NOT here. Three are refused for
# a reason that predates this repair and survives it -- printing an Optional
# field without narrowing it, storing into one Optional field after a branch
# narrowed another, and two Optional fields tested in one expression -- and the
# fourth compiles and LEAKS: overwriting a `str | None` field drops the string
# it replaces (41 B, measured identical on the binary before this repair, so it
# is exposed here rather than caused here). That is why `name` below is never
# reassigned while `items` and `count` are. All four are the inline union field
# storage, and all four are measured in
# tests/probe/wb_optional_field_inline_storage.py.
#
# Every expected line is python3.14's.


class Holder:
    def __init__(self) -> None:
        self.name: str | None = "a"
        self.items: list[int] | None = None
        self.raw: bytes | None = b"xy"
        self.count: int | None = None


h = Holder()

# --- `is None` directly on the field read ---------------------------------
print(h.name is None, h.items is None, h.raw is None, h.count is None)
print(h.name is not None, h.items is not None)

# --- assigned, then tested -------------------------------------------------
h.items = [1, 2, 3]
h.count = 7
print(h.name is None, h.items is None, h.count is None)

# --- read into a local, narrow, use ---------------------------------------
got = h.items
if got is not None:
    print(len(got), got[0], got[2])

# --- back to None, and the narrowing must not fire ------------------------
h.items = None
again = h.items
if again is None:
    print("cleared")
else:
    print("wrong")


# --- a bytes member, which is two lanes like str --------------------------
class Raw:
    def __init__(self) -> None:
        self.blob: bytes | None = b"xyz"


r = Raw()
print(r.blob is None)
blob = r.blob
if blob is not None:
    print(len(blob))


# --- a str member read out of a fresh instance, then narrowed -------------
class Named:
    def __init__(self, n: str | None) -> None:
        self.n: str | None = n


print(Named("q").n is None, Named(None).n is None)
picked = Named("q").n
if picked is not None:
    print(picked, len(picked))
