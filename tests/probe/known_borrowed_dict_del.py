# probe: REPORTED passes: dict del on a borrowed (parameter) dict (in place)
# axes: acquire=param width=w1dict op=del flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 1

def drop(d: dict[str, int]) -> None:
    del d["a"]


d: dict[str, int] = {"a": 1, "b": 2}
drop(d)
print(len(d))
