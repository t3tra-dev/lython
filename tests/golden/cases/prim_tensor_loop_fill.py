from lyrt import from_prim
from lyrt.prim import Float, Int, Matrix

# ループ内の要素代入で行列を埋める (ランタイム添字 + loop-carried setitem)

m = Matrix[Float[32], 3, 3].zeros()
v = 0.0
for i in range(3):
    for j in range(3):
        m[i, j] = v
        v = v + 1.0
print(from_prim(m))

# 対角の設定
d = Matrix[Int[32], 4, 4].zeros()
for k in range(4):
    d[k, k] = 1
print(from_prim(d[0, 0]))
print(from_prim(d[3, 3]))
print(from_prim(d[0, 3]))

# 演算と要素代入の混合
a = Matrix[Float[32], 2, 2].ones()
for t in range(3):
    a = a * 2.0
    a[0, 0] = 0.0
print(from_prim(a))
