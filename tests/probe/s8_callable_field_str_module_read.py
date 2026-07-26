# probe: 同じ str コールアブルフィールドを、メソッドを介さずモジュール直下で
#   読み戻して呼ぶ。同じ assignability 診断になる = 読み出し位置は関係ない。
#   (出典: kernel-sidedefects)
# axes: width=callable(str) op=field flow=straight scope=module
# CLASSIFICATION @ kernel/4b fa71a3c: 3 loud 拒否 (診断)
#   attribute value '!py.contract<"builtins.function">' is not assignable to field '!py.callable<[], returns = [!py.contract<"builtins.str">]>'
# CPython 3.14 expects: hi

from typing import Callable


def make() -> str:
    return "hi"


class Holder:
    def __init__(self, f: Callable[[], str]) -> None:
        self._f: Callable[[], str] = f


h = Holder(make)
g: Callable[[], str] = h._f
print(g())
