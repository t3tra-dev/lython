# What: an unannotated parameter with a default has a complete static
# description in the default itself, so the omitted-argument call has to
# produce that value -- and a call that supplies an argument of another type
# has to widen the parameter rather than replace it. Only running it shows
# which value each call actually bound.
def plain(x=3):
    return x


print(plain(), plain(5))


def text(s="hi"):
    return s + "!"


print(text(), text("yo"))


def flagged(on=True):
    return 1 if on else 0


print(flagged(), flagged(False))


def optional(n=None):
    if n is None:
        return -1
    return n


print(optional(), optional(7))


def widened(v=3):
    return v


print(widened("a"), widened())


def keyword_only(*, k=2.5):
    return k


print(keyword_only(), keyword_only(k=0.5))

square = lambda n=4: n * n
print(square(), square(3))
