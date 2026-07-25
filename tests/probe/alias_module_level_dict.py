# probe: 同じ「フィールドをローカルに読み出して挿入し書き戻す」形を、メソッドを
#   経由せずモジュール直下で書く。この綴りだけ abort ではなく **silent** になる
#   (挿入が黙って失われる) ので、alias staleness の silent の顔にあたる。
# axes: acquire=inline width=w1dict(box-fronted) op=alias-read+insert+writeback flow=straight scope=module
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 1


class Node:
    def __init__(self, v: dict[str, int]) -> None:
        self._f: dict[str, int] = v


v0: dict[str, int] = {}
n = Node(v0)
d: dict[str, int] = n._f
d["x"] = 1
n._f = d
print(len(n._f))
