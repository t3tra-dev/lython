# What: a def declared in a loop body is a name the empty container's element
# scan has to know the type of, and the closure it makes reads the frame's
# cell -- only calling the elements shows both, since each answer differs.
def collect() -> "list[int]":
    fs = []
    for i in range(3):
        def step() -> int:
            return i * 5
        fs.append(step)
    return [f() for f in fs]


print(collect())

pending = []
for word in ["ab", "cde"]:
    def measure() -> int:
        return len(word)
    pending.append(measure)
print([f() for f in pending])


def mixed() -> "list[str]":
    out = []
    for tag in ["x", "y"]:
        def label() -> str:
            return tag + "!"
        out.append(label)
        out.append(lambda: tag * 2)
    return [f() for f in out]


print(mixed())
