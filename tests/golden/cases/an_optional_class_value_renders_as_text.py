# What: a `type[BaseException] | None` value renders through print, str, repr,
# f-string conversions, %-formatting and ascii as None or as `<class ...>`.
# Why run: which member the value holds is a run-time fact.
class Mine(Exception):
    pass


def f(u: type[BaseException] | None) -> None:
    print(u)
    print(str(u), repr(u), f"{u}|{u!r}|{u!a}", "%s %r" % (u, u), ascii(u))


f(None)
f(KeyError)
f(Mine)
g: type[BaseException] | None = None
print(g)
g = KeyError
print(g)
