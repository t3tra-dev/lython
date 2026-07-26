# **1 つの store 本体を 6 通りの位置から到達させ、全部同じ答えになることを押さえる。**
#
# これは RFC (`rfc/object-ownership-kernel.md` §1.3「正しさが inline に依存している」)
# の表そのものである。段階 4a より前、**同一のメソッド本体**が呼び方だけで結果を
# 変えていた:
#
#   直線で 1 回呼ぶ                 → 正しい
#   while / for から呼ぶ            → MLIR dominance 失敗 (Lython の診断ではない)
#   コンテナ要素をレシーバにする     → silent (要素側もローカル側も旧値)
#   同じ本体を通常関数にする         → silent (callee は渡された値に何も書かない)
#
# メソッドは standalone シンボルを持たず常に呼び出し側へ展開されるので、「正しい」
# ケースが正しかったのは **inline されたから**にすぎなかった。つまり正しさが
# 最適化判断の副作用であって契約ではなかった。フィールド store が安定した heap
# スロットへの store になった今、書き込み先はどの位置から見ても同じ 1 箇所なので、
# 呼び方は結果に影響しない。
#
# 個別の probe (`flow_while_*` / `op_store_into_*` / `wb_param_*`) がそれぞれの
# 位置を押さえているが、**「同じ本体が位置によらず同じ答えを出す」ことを 1 本の
# プログラムで主張しているのはこのファイルだけ**である。30 件の probe の連言から
# 推論するのと、直接測るのは別のことなので分けて置く。
class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v

    # メソッド形 (常に inline される)
    def set_method(self, x: float) -> None:
        self.f = x


# 同じ本体を通常関数にしたもの (inline されない)
def set_plain(b: Box, x: float) -> None:
    b.f = x


def mk() -> Box:
    z: float = 0.0
    return Box(z)


# 1. 直線でメソッドを 1 回呼ぶ
a = mk()
a.set_method(1.5)
print(a.f)

# 2. ループからメソッドを呼ぶ (旧: dominance 失敗)
b = mk()
i: int = 0
while i < 3:
    b.set_method(2.5)
    i = i + 1
print(b.f)

# 3. コンテナ要素をレシーバにする (旧: silent。要素とローカルが同じ 1 個であること
#    も同時に押さえる)
c = mk()
xs: list[Box] = [c]
xs[0].set_method(3.5)
print(xs[0].f, c.f)

# 4. 同じ本体を通常関数として呼ぶ (旧: silent = N1)
d = mk()
set_plain(d, 4.5)
print(d.f)

# 5. 分岐の片アームから (旧: dominance 失敗)
e = mk()
if len(xs) > 0:
    e.set_method(5.5)
print(e.f)

# 6. 通常関数をループから (旧: silent と dominance 失敗の組み合わせ)
g = mk()
j: int = 0
while j < 2:
    set_plain(g, 6.5)
    j = j + 1
print(g.f)
