# WHAT: a method's **kwargs holds the keywords the call gave in the order it
# gave them -- the order a function's does, and CPython's -- whether the call
# spells them out or forwards another method's **kwargs.
#
# WHY THIS IS RUN: the order is the dict the run builds.


class Recorder:
    def keys(self, **kw: int) -> list[str]:
        return list(kw.keys())

    def relay(self, **kw: int) -> list[str]:
        return self.keys(**kw)


def keys(**kw: int) -> list[str]:
    return list(kw.keys())


r = Recorder()
print(r.keys(zeta=1, alpha=2, mid=3, beta=4))
print(keys(zeta=1, alpha=2, mid=3, beta=4))
print(r.relay(omega=1, beta=2, alpha=3))
