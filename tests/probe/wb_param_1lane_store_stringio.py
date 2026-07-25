# probe: N1 の store 形を 1 レーン header-fronted contract で行う -- 通常関数が
#   借用レシーバのフィールドを別の io.StringIO に「再束縛」し、呼び出し元が読む。
#   1 レーンでも silent になるので「1 レーンなら境界を越える」は成り立たない。
# axes: acquire=param width=1lane-header-fronted op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: fresh

import io


class Sink:
    def __init__(self, buf: io.StringIO) -> None:
        self._buf: io.StringIO = buf


def make() -> Sink:
    return Sink(io.StringIO())


def replace(s: Sink) -> None:
    other = io.StringIO()
    other.write("fresh")
    s._buf = other


n = make()
n._buf.write("old")
replace(n)
print(n._buf.getvalue())
