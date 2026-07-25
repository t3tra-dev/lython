# probe: `defaultdict` の形 -- int を返す factory コールアブルのフィールドと
#   dict フィールドを同じクラスに持つ。int 版と同じ prim_i64 診断。
#   collections.defaultdict の移植がこの形に当たるため、実務上の重みが最も大きい。
#   (出典: kernel-sidedefects)
# axes: width=callable(int)+dict op=field flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 3 loud 拒否 (診断)
#   function target 'zero__lyrt_prim_i64' returned too few values for result object ABI
# CPython 3.14 expects: 0

from typing import Callable


def zero() -> int:
    return 0


class DefaultBox:
    def __init__(self, factory: Callable[[], int]) -> None:
        self.factory: Callable[[], int] = factory
        self.store: dict[str, int] = {}

    def get(self, k: str) -> int:
        if k in self.store:
            return self.store[k]
        f: Callable[[], int] = self.factory
        v: int = f()
        self.store[k] = v
        return v


d = DefaultBox(zero)
print(d.get("a"))
