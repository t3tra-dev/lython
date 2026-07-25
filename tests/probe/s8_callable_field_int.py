# probe: **int** を返すコールアブルをフィールドに格納する。str 版とは
#   **別の診断**になる (`function target 'make__lyrt_prim_i64' returned too few
#   values for result object ABI`) = S8 は単一の欠陥ではない。戻り値型が
#   native int 経路に落ちるかどうかで診断が変わる。
#   (出典: kernel-sidedefects)
# axes: width=callable(int) op=field flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 3 loud 拒否 (診断)
#   function target 'make__lyrt_prim_i64' returned too few values for result object ABI
# CPython 3.14 expects: 7

from typing import Callable


def make() -> int:
    return 7


class Holder:
    def __init__(self, f: Callable[[], int]) -> None:
        self._f: Callable[[], int] = f

    def call(self) -> int:
        g: Callable[[], int] = self._f
        return g()


h = Holder(make)
print(h.call())
