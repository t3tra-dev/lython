from lyrt import from_prim
from lyrt.prim import Float, Matrix, Vector

# 除算は float 要素限定の elementwise linalg 演算として emit される

a = Matrix[Float[32], 2, 2]([[1.0, 2.0], [3.0, 4.0]])
b = Matrix[Float[32], 2, 2]([[2.0, 4.0], [8.0, 16.0]])
print(from_prim(a / b))

v = Vector[Float[64], 3]([1.0, 2.0, 3.0])
w = Vector[Float[64], 3]([4.0, 4.0, 4.0])
print(from_prim(v / w))

# スカラー prim の除算も同じ表面
s = Float[32](3.0)
t = Float[32](2.0)
print(from_prim(s / t))

# ループ内で生成してループ内で消費する (ベンチマークの基本形)
acc = 0.0
for k in range(3):
    c = a / b
    acc = acc + from_prim(c[0, 0])
print(acc)
