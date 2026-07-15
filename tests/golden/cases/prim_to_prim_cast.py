from lyrt import from_prim, to_prim
from lyrt.prim import Float, Int, Matrix

# dtype 混在の演算は拒否されたまま、キャストは to_prim で明示する。
# tensor 間キャストは float <-> float, int -> float, int 拡幅のみ
# (float -> int と int 縮小は要素ごとに trap できないので静的に拒否)

a = Matrix[Float[32], 2, 2]([[1.5, 2.5], [3.5, 4.5]])
b = Matrix[Float[64], 2, 2].full(2.0)
c = to_prim(a, Matrix[Float[64], 2, 2]) + b
print(from_prim(c))

d = to_prim(b, Matrix[Float[32], 2, 2])
print(from_prim(d * a))

i = Matrix[Int[32], 2, 2]([[1, 2], [3, 4]])
f = to_prim(i, Matrix[Float[32], 2, 2])
print(from_prim(f / 2.0))

w = to_prim(i, Matrix[Int[64], 2, 2])
print(from_prim(w + Matrix[Int[64], 2, 2].ones()))

# スカラーの to_prim: Python 値 / prim 値から
s = to_prim(3, Int[32])
print(from_prim(s))
t = to_prim(2.5, Float[64])
print(from_prim(t))
u = to_prim(Float[64](1.25), Float[32])
print(from_prim(u))

# 行列積の dtype 揃えパターン
g = to_prim(i, Matrix[Float[32], 2, 2]) @ a
print(from_prim(g[0, 0]))
