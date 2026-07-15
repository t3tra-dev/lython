from lyrt import from_prim
from lyrt.prim import Float, Int, Matrix, Tensor, Vector

# elementwise 演算は linalg 構造化演算として emit される: リテラル入力でも
# 演算が定数畳み込みで消えず、実行時のループとして残る

# 整数要素の add / sub / mul
ia = Matrix[Int[32], 2, 2]([[1, 2], [3, 4]])
ib = Matrix[Int[32], 2, 2]([[5, 6], [7, 8]])
print(from_prim(ia + ib))
print(from_prim(ia - ib))
print(from_prim(ia * ib))

# 浮動小数点要素
fa = Vector[Float[64], 3]([1.0, 2.0, 3.0])
fb = Vector[Float[64], 3]([0.5, 0.25, 0.125])
print(from_prim(fa + fb))
print(from_prim(fa * fb))

# rank 3 でも同じ経路
t = Tensor[Int[64], 2, 2, 2]([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
u = Tensor[Int[64], 2, 2, 2].full(10)
print(from_prim(t * u))

# ループ内で生成してループ内で消費する (ベンチマークの基本形)
s = 0.0
for k in range(3):
    c = fa * fb
    s = s + from_prim(c[0])
print(s)
