# What: a generator that yields a container it also still binds needs one
# reference per holder -- the consumer's and the suspended frame's. Only
# running it shows the count was right: one short and a later trip frees a
# list the frame is still using, one long and it never frees at all.
def steps(values: "list[int]"):
    current: list[int] = []
    for v in values:
        current.append(v)
        yield current
        current = []


def running(values: "list[int]"):
    seen: list[int] = []
    for v in values:
        seen.append(v)
        yield seen


for step in steps([1, 2, 3]):
    print(step)
for snapshot in running([7, 8]):
    print(snapshot)
