from lyrt import from_prim
from lyrt.prim import Float, Int, Tensor

# rank>=3 の @ は等 rank・先頭次元一致のバッチ行列積 (末尾 2 次元で契約)

a = Tensor[Float[32], 2, 2, 3](
    [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ]
)
b = Tensor[Float[32], 2, 3, 2](
    [
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
    ]
)
c = a @ b
print(from_prim(c))

# rank-4 は先頭次元を batch に collapse して同じ契約に乗る
d = Tensor[Int[32], 2, 2, 2, 2].full(2)
e = Tensor[Int[32], 2, 2, 2, 2].full(3)
f = d @ e
print(from_prim(f[0, 0, 0, 0]))
print(from_prim(f[1, 1, 1, 1]))

# ループ内で生成してループ内で消費する (ベンチマークの基本形)
acc = 0.0
for k in range(3):
    g = a @ b
    acc = acc + from_prim(g[0, 0, 0])
print(acc)
