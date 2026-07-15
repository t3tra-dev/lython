from lyrt import from_prim
from lyrt.prim import Float, Int, Matrix, Tensor, Vector

# zeros / ones は 3 つの shaped primitive すべてで使える
print(from_prim(Vector[Float[32], 3].zeros()))
print(from_prim(Vector[Float[32], 3].ones()))
print(from_prim(Matrix[Float[32], 2, 2].zeros()))
print(from_prim(Matrix[Float[32], 2, 2].ones()))
print(from_prim(Tensor[Float[32], 2, 2, 2].ones()))

# 整数要素
print(from_prim(Tensor[Int[32], 2, 2].zeros()))
print(from_prim(Vector[Int[64], 3].ones()))

# full: 数値リテラル
print(from_prim(Matrix[Float[32], 2, 2].full(2.5)))
print(from_prim(Vector[Int[32], 3].full(7)))

# full: 実行時に決まる primitive スカラー
x = Float[32](1.5) + Float[32](2.0)
print(from_prim(Matrix[Float[32], 2, 2].full(x)))

# 生成した値はそのまま演算・添字に使える
m = Matrix[Float[32], 2, 2].ones()
n = Matrix[Float[32], 2, 2].full(3.0)
print(from_prim(m + n))
print(from_prim(m @ n))
print(from_prim(n[1, 1]))
