from lyrt import from_prim
from lyrt.prim import Float, Int, Matrix, Vector

# 実行時の Python 数値は py.cast.to_prim 経由で primitive に注入できる

# スカラーコンストラクタ: py float / py int から
x = 1.5
y = x + 0.75
print(from_prim(Float[32](y)))
n = 40
print(from_prim(Int[32](n + 2)))
print(from_prim(Float[64](n + 2)))

# full: 実行時スカラーで splat
print(from_prim(Matrix[Float[32], 2, 2].full(y * 2.0)))

# コンストラクタ: ネストはリテラル構造のまま、リーフだけ実行時値を許す
a = 1.5
b = a + 1.0
m = Matrix[Float[32], 2, 2]([[a, b], [b * 2.0, 4.0]])
print(from_prim(m))
k = 7
print(from_prim(Vector[Int[64], 3]([k, k + 1, 100])))

# 要素代入: 実行時値を静的座標へ書き込む
w = Matrix[Float[32], 2, 2].zeros()
w[0, 1] = x
w[1, 0] = x * 2.0
print(from_prim(w))

# 注入したデータはそのまま matmul に流せる
c = m @ w
print(from_prim(c[0, 0]))
print(from_prim(c[1, 1]))
