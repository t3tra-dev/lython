# probe: S8 の基準線 -- コールアブルをローカルに束縛して呼ぶ (フィールドなし)。
#   通る。S8 が「コールアブルの表現が無い」問題ではないことを示す 1 件目。
#   (出典: kernel-sidedefects の 7 形状マトリクス)
# axes: width=callable op=local-bind flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: hi

from typing import Callable


def make() -> str:
    return "hi"


g: Callable[[], str] = make
print(g())
