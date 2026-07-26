# probe: **同一の store 本体**を 6 つの位置から到達させ、6 つの答えが一致するか。
#   本文書はこれを「~30 probe の連言」から推論していたが、**連言は主張であって
#   測定ではない**: 「各位置がそれぞれの本体で通る」は
#   「1 つの本体が 6 位置で同じ答えを出す」と同じ言明ではない。
#   RFC が実際に問題にしているのは後者なので、直接測る形を置く。
#   (k-4a の指摘。同トラックは golden 側に
#   `cases/class_field_store_inline_independent` を置いた — **`kernel/4a` 側に
#   あり、このブランチには無い**ので、統合後に参照が解決する。こちらは corpus 側の
#   独立実装 — 相手の測定を検証するときは同じものを走らせないため。)
#
#   `c3de5e7` 時点で各位置が何をしていたか (本文 §「3 経路表」より):
#     直線・メソッド      -> 正しい (ただし inline されていたから)
#     通常関数            -> silent (store が痕跡なく消える)
#     ループ内            -> MLIR dominance 失敗
#     コンテナ要素経由    -> silent (要素と元のローカルが食い違う)
#     別オブジェクトの
#     フィールド経由      -> abort (二重解放)
#
#   コンテナの行は `xs[0].f` と `c.f` を**同じ print で両方**主張するので、
#   「要素と元のローカルが偶然一致した 2 つの物」ではなく**同一物**であることを
#   押さえる (N3 と N4 は別 probe だったので、その連言を 1 観測にする)。
# axes: width=w2float op=one-body-six-positions flow=mixed observe=inline-independence
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 1.5 / 2.5 / 3.5 3.5 / 4.5 / 5.5 / 6.5


class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v

    def store(self, v: float) -> None:
        self.f = v


def store_fn(b: Box, v: float) -> None:
    b.f = v


def mk() -> Box:
    return Box(0.0)


class Holder:
    def __init__(self, b: Box) -> None:
        self.inner: Box = b


# 1. 直線、呼び出し由来レシーバ、メソッド本体。
a = mk()
a.store(1.5)
print(a.f)

# 2. 同じ本体をループから呼ぶ。
b = mk()
i = 0
while i < 2:
    b.store(2.5)
    i = i + 1
print(b.f)

# 3. コンテナ要素経由。要素とローカルの両方を同じ print で主張する。
c = mk()
xs: list[Box] = [c]
xs[0].store(3.5)
print(xs[0].f, c.f)

# 4. 別オブジェクトのフィールド経由。
h = Holder(mk())
h.inner.store(4.5)
print(h.inner.f)

# 5. **同一の本体を通常関数として**呼ぶ (メソッドではない = inline されない経路)。
e = mk()
store_fn(e, 5.5)
print(e.f)

# 6. 2 段の呼び出しを跨ぐ。
def outer(b: Box, v: float) -> None:
    store_fn(b, v)


g = mk()
outer(g, 6.5)
print(g.f)
