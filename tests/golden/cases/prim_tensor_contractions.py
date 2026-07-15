from lyrt import from_prim
from lyrt.prim import Float, Int, Matrix, Vector

# @ はオペランドの rank で契約を選ぶ (numpy 準拠):
# 1x1 -> dot (スカラー), 2x1 -> matvec, 1x2 -> vecmat

v = Vector[Float[32], 3]([1.0, 2.0, 3.0])
w = Vector[Float[32], 3]([4.0, 5.0, 6.0])
print(from_prim(v @ w))

m = Matrix[Float[32], 2, 3]([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
print(from_prim(m @ v))
print(from_prim(v @ Matrix[Float[32], 3, 2]([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])))

# 整数要素の dot
iv = Vector[Int[32], 3]([1, 2, 3])
iw = Vector[Int[32], 3]([4, 5, 6])
print(from_prim(iv @ iw))

# matvec 結果は Vector としてそのまま演算に使える
u = m @ v
print(from_prim(u * 2.0))

# ループ内で生成してループ内で消費する (ベンチマークの基本形)
acc = 0.0
for k in range(3):
    d = v @ w
    acc = acc + from_prim(d)
print(acc)
