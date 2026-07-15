from lyrt import from_prim
from lyrt.prim import Float, Int, Matrix

# スカラーは tensor 側の shape に splat されて elementwise 経路に乗る。
# 左右どちらのオペランドでもよい

m = Matrix[Float[32], 2, 2]([[1.0, 2.0], [3.0, 4.0]])
print(from_prim(m * 2.0))
print(from_prim(2.0 * m))
print(from_prim(m + 1.0))
print(from_prim(1.0 - m))
print(from_prim(m / 2.0))
print(from_prim(8.0 / m))

# int リテラルは float 要素へ昇格して broadcast できる
print(from_prim(m + 1))

# prim スカラーも broadcast できる
s = Float[32](0.5)
print(from_prim(m * s))

# 整数要素には整数スカラーのみ
i = Matrix[Int[32], 2, 2]([[1, 2], [3, 4]])
print(from_prim(i * 3))
print(from_prim(10 - i))
k = Int[32](2)
print(from_prim(i * k))

# ループ内で生成してループ内で消費する (ベンチマークの基本形)
acc = 0.0
for n in range(3):
    c = m * 2.0
    acc = acc + from_prim(c[0, 0])
print(acc)
