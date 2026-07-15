from lyrt import from_prim
from lyrt.prim import Float, Int, Matrix, Tensor, Vector

# 要素の読み出しは静的な形状座標で行う
m = Matrix[Float[32], 2, 3]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(from_prim(m[0, 0]))
print(from_prim(m[1, 2]))

# 負のインデックスは末尾からの参照
print(from_prim(m[-1, -1]))

# rank 1 は単一インデックス、rank 3 は 3 つ
v = Vector[Float[32], 3]([1.5, 2.5, 3.5])
print(from_prim(v[1]))
t3 = Tensor[Int[32], 2, 2, 2]([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(from_prim(t3[1, 0, 1]))

# 整数要素の tensor
t = Tensor[Int[32], 2, 2]([[10, 20], [30, 40]])
print(from_prim(t[1, 0]))

# 部分代入は値意味論: 代入後の local が新しい値を観測する
m[0, 1] = Float[32](9.0)
print(from_prim(m))
print(from_prim(m[0, 1]))

# 代入前の値をコピーした別の local は影響を受けない
before = Matrix[Float[32], 2, 3]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
after = before
after[0, 0] = Float[32](7.0)
print(from_prim(before[0, 0]))
print(from_prim(after[0, 0]))

# matmul の結果に対する要素アクセス
a = Matrix[Float[32], 2, 3]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = Matrix[Float[32], 3, 2]([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
c = a @ b
print(from_prim(c[0, 0]))
print(from_prim(c[1, 1]))
