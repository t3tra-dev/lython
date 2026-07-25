# probe: callee stores into and reads back a borrowed receiver's bytes field, then the caller reads it
# axes: acquire=param width=bytes op=rebind flow=straight observe=writeback-both
# CLASSIFICATION: 2 silent 誤実行
#   cpython="in callee: b'abcd'\nin caller: b'abcd'\n" lyc="in callee: b'abcd'\nin caller: b''\n"
# CPython 3.14 expects: in callee: b'abcd' / in caller: b'abcd'

class Box:
    def __init__(self, v: bytes) -> None:
        self.f: bytes = v


def mk() -> Box:
    v: bytes = b""
    return Box(v)


def rebind(b: Box) -> None:
    fresh: bytes = b"abcd"
    b.f = fresh
    o = b
    print("in callee:", o.f)


o = mk()
rebind(o)
print("in caller:", o.f)
