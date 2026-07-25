# probe: N1 の形を 1 レーン header-fronted contract で行う -- 通常関数が借用
#   レシーバのフィールド (io.StringIO) を「読み出して in-place 変異」する。
#   header 経由なら関数境界を越えるはず (3 経路表の 4 行目の確認)。
# axes: acquire=param width=1lane-header-fronted op=inplace-mutate flow=straight observe=writeback
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: xx

import io


class Sink:
    def __init__(self, buf: io.StringIO) -> None:
        self._buf: io.StringIO = buf


def make() -> Sink:
    return Sink(io.StringIO())


def emit(s: Sink) -> None:
    s._buf.write("x")


n = make()
emit(n)
emit(n)
print(n._buf.getvalue())
