# What: `lambda i=i:` -- the idiom Python has for capturing a loop variable by
# VALUE, beside the plain `lambda: i` that captures the binding. Both are here
# because they must disagree: the frozen one answers what each trip held, and
# the plain one answers what the name holds when it is called. Calling every
# function and printing the answers side by side is the only way to see that.
frozen = []
live = []
for i in range(3):
    frozen.append(lambda i=i: i * 2)
    live.append(lambda: i * 2)

print([f() for f in frozen])
print([f() for f in live])

# The default is still a default: passing an argument overrides it.
print(frozen[0](), frozen[0](9), frozen[2]())

names = ["ab", "cd"]
readers = []
for name in names:
    readers.append(lambda name=name: name.upper())
print([r() for r in readers])

# A default that is an EXPRESSION over the loop variable is evaluated once per
# trip, in the trip's frame.
scaled = []
for i in range(3):
    scaled.append(lambda v=i * 10: v + 1)
print([s() for s in scaled])


# Inside a function, where the loop variable is a local rather than a global.
def build() -> "list[int]":
    fs = []
    for i in range(3):
        fs.append(lambda i=i: i + 1)
    return [f() for f in fs]


print(build())


# The `def` spelling of the same thing, which has always worked, so the two
# stay measured against each other.
defs = []
for i in range(3):
    def make(i: int = i) -> int:
        return i * 2
    defs.append(make)
print([f() for f in defs])
