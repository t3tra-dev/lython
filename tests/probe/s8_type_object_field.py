# probe: 型オブジェクトをフィールドに格納する。コールアブル 2 種とはさらに
#   **別の診断** (`callable ABI type has no concrete runtime contract:
#   '!py.type<...>'`) = S8 は 3 つの診断に分かれる。
#   (出典: kernel-sidedefects。私の既存 known_field_type_object.py は
#   `type[...]` を呼ぶところまで含む変種)
# axes: width=type-object op=field flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   callable ABI type has no concrete runtime contract: '!py.type<!py.contract<"Inner">>'
# CPython 3.14 expects: 1

class Inner:
    def __init__(self, v: int) -> None:
        self._v: int = v


class Holder:
    def __init__(self, cls: type[Inner]) -> None:
        self._cls: type[Inner] = cls


h = Holder(Inner)
print(1)
