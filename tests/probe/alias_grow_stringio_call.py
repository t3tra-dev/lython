# probe: 照会 A/B の対照 -- 完全に header-fronted な 1 レーン contract
#   (io.StringIO) のフィールドを同じ形で変異させる。657f0d8 でも正常終了した。
# axes: acquire=call width=1lane-header-fronted op=alias-grow flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: abcdef
# 観測: libgmalloc: 正常終了

import io


class Sink:
    def __init__(self, buf: io.StringIO) -> None:
        self._buf: io.StringIO = buf

    def add(self, s: str) -> None:
        b: io.StringIO = self._buf
        b.write(s)


def make() -> Sink:
    return Sink(io.StringIO())


n = make()
n.add("ab")
n.add("cd")
n.add("ef")
print(n._buf.getvalue())
