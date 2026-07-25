# probe: str を返すコールアブルをクラスフィールドに格納し、メソッド内でローカルに
#   読み戻して呼ぶ。拒否される。診断は assignability
#   (`builtins.function` is not assignable to field `!py.callable<...>`)。
#   上の list 要素版が通ることと対比すること。
#   (出典: kernel-sidedefects)
# axes: width=callable(str) op=field flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   attribute value '!py.contract<"builtins.function">' is not assignable to field '!py.callable<[], returns = [!py.contract<"builtins.str">]>'
# CPython 3.14 expects: hi

from typing import Callable


def make() -> str:
    return "hi"


class Holder:
    def __init__(self, f: Callable[[], str]) -> None:
        self._f: Callable[[], str] = f

    def call(self) -> str:
        g: Callable[[], str] = self._f
        return g()


h = Holder(make)
print(h.call())
