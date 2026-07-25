# dataclass のフィールドを、全ての物理形で読み書きする。CPython の出力と完全
# 一致しなければならない。
#
# これは lowerInit のフィールド記録経路で、`__init__` を書いたクラスが通る
# attr.set 経路とは別物である。int/bool フィールドの値はインスタンスヘッダの
# ワードに入るが、この経路は**プレースホルダのレーンに書いていた**ため、
# ヘッダのワードを読む attr.get は常に 0 を返していた — 診断なし、カバレッジ
# なしの silent (`P(3, ...)` の直後に `p.x` が 0)。既存の dataclass golden が
# 捕まえなかったのは、どれも比較・repr・等値だけを見ていて、フィールドを
# 個別に読む形が無かったためである。
#
# bool は同じワード機構を使う。参照型フィールド (float / str / list / ユーザー
# クラス) は box16 スロット経路なので、別フレームからの再束縛が呼び出し元に
# 見えることまで同時に押さえる。
from dataclasses import dataclass


class Inner:
    def __init__(self, n: int) -> None:
        self.n: int = n


# repr/eq を止めているのは、合成される `__repr__` / `__eq__` がユーザークラス型
# フィールドをまだ扱えないため (別根、§1.7 の型システム側)。ここで測りたいのは
# フィールド 1 本ずつの読み書きなので、合成メソッドは要らない。
@dataclass(repr=False, eq=False)
class P:
    x: int
    flag: bool
    ratio: float
    name: str
    xs: list[int]
    inner: Inner


def bump(p: P) -> None:
    p.x = 9
    p.flag = False
    p.ratio = 0.5
    p.name = "after"
    fresh: list[int] = [7, 8, 9]
    p.xs = fresh
    p.inner = Inner(42)


p = P(3, True, 2.5, "before", [1, 2], Inner(1))
print(p.x, p.flag, p.ratio, p.name, len(p.xs), p.inner.n)
bump(p)
print(p.x, p.flag, p.ratio, p.name, len(p.xs), p.inner.n)
