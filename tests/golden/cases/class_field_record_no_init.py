# `__init__` を書かないクラス (field-record 構築規則) のフィールドを、6 つの物理形
# すべてで読み書きする。
#
# **これは `class_noinit_field_record` (dataclass 版) とは別の経路である。** dataclass は
# `__init__` を合成するので attr.set 経路を通る。`__init__` を書かないこの形だけが
# `lowerInit` のフィールド記録経路に入り、そこは int/bool の**ヘッダワードを書いて
# いなかった** — プレースホルダのレーンに書き、attr.get は常にワードを読むので、
# `Rec(3, ...)` の直後に `r.x` が 0 になった。診断なし、カバレッジなしの silent。
#
# **dataclass 版はこの欠陥を検出しない** (故障注入で確認済み: `lowerInit` の
# ヘッダワード store を無効化しても dataclass 版は正しい値を出し、この形だけが
# 0xAAAA... を出す)。修正の回帰テストはこのファイルである。
#
# 期待値は CPython 3.14 のもので、明示 `__init__` を持つ参照綴りから生成した
# (CPython は引数付きの裸クラス構築を受け付けないので、この綴り自体は動かない)。
# 参照綴りは `bump` までこのファイルと同一で、`__init__` があるだけである。
class Inner:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Rec:
    x: int
    flag: bool
    ratio: float
    name: str
    xs: list[int]
    inner: Inner


def bump(r: Rec) -> None:
    r.x = 9
    r.flag = False
    r.ratio = 0.5
    r.name = "after"
    fresh: list[int] = [7, 8, 9]
    r.xs = fresh
    r.inner = Inner(42)


r = Rec(3, True, 2.5, "before", [1, 2], Inner(1))
print(r.x, r.flag, r.ratio, r.name, len(r.xs), r.inner.n)
bump(r)
print(r.x, r.flag, r.ratio, r.name, len(r.xs), r.inner.n)
