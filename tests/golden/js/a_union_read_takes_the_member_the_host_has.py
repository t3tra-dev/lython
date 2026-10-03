# What: a member the stub types `str | None` reads as whichever the host
# returns, and narrows like any union; `X.new(...)` constructs.
#
# Why run: which member the value is is decided by the host at run time.
from js import URLSearchParams

params = URLSearchParams.new("a=1&b=two")
present = params.get("a")
absent = params.get("zzz")
print(present, absent, params.has("b"), params.size)
if present is not None:
    print("narrowed:", present + "!")
params.append("c", "3")
print(params.toString())
