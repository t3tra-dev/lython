# A class body is a SCOPE, and CPython runs it with each assignment visible to
# the ones after it. Nothing bound those names here, so a later attribute could
# not be typed from an earlier one -- it fell off the slot channel onto the
# constant channel, which has no arm for an expression and said so from the
# lowering.

WIDTH = 10


class Layout:
    width = 80
    half = width // 2
    label = "w" + str(width)
    both = half + width
    scaled = width * WIDTH


print(Layout.width, Layout.half, Layout.label, Layout.both, Layout.scaled)


class Annotated:
    n: int = 3
    m: int = n + 1


print(Annotated.n, Annotated.m)


class Derived:
    names = ["a", "b"]
    first = names[0]
    count = len(names)


print(Derived.first, Derived.count, Derived.names)


# The body may rebind a name, and each reader sees what was bound at ITS line.
class Rebinds:
    n = 1
    m = n
    n = 2


print(Rebinds.n, Rebinds.m)


# And the names do not outlive the body: a module global of the same spelling is
# itself again below.
n = 100


class Shadows:
    n = 1
    m = n + 1


print(Shadows.n, Shadows.m, n)
