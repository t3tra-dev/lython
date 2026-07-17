# A raise inside a resumed generator shows the CPython frames: the for line
# anchored under the iterator expression, the generator body frame under its
# Python name (no state-machine clone suffixes or plumbing frames).
def g():
    yield 1
    raise ValueError("in gen")


for v in g():
    print(v)
