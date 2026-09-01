# What: a dict walk inside a generator goes through the keys, and CPython
# checks the container's size in every `__next__` -- including the call that
# finds it exhausted. Only running it shows the RuntimeError arrives on the
# step after the write, rather than the walk quietly finishing over a copy.
def grow(d: "dict[str, int]"):
    for k in d:
        d[k + "!"] = 9
        yield k


print(list(grow({"a": 1})))
