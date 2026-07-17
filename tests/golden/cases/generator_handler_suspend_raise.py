# A generator suspended INSIDE an exception handler must not keep the
# process's single EH token occupied: the suspension stashes the in-flight
# token with the generator frame, so user raises afterwards work, and a
# later drop of the still-suspended generator finalizes without trapping.
def swallower():
    try:
        yield 1
    except GeneratorExit:
        yield 2


g = swallower()
print(next(g))
try:
    g.close()
except RuntimeError as e:
    print("RuntimeError:", e)
try:
    raise ValueError("after handler suspend")
except ValueError as e:
    print("caught:", e)
print("done")
