from lyrt import from_prim
from lyrt.prim import Float, Matrix, Vector

# loop-carried な shaped primitive: 再代入はループ前に確保された 1 つの
# バッファへの書き戻しとして lowering される (反復ごとの確保なし)

# 累積 (元の拒否ケース)
a = Matrix[Float[32], 2, 2]([[1.0, 2.0], [3.0, 4.0]])
da = Matrix[Float[32], 2, 2]([[0.5, 0.5], [0.5, 0.5]])
i = 0
while i < 3:
    a = a + da
    i = i + 1
print(from_prim(a))

# carried 値を縮約の入力に使う (in-place では成立しないケース)
A = Matrix[Float[32], 2, 2]([[0.0, 1.0], [1.0, 0.0]])
x = Matrix[Float[32], 2, 1]([[1.0], [2.0]])
for k in range(3):
    x = A @ x
print(from_prim(x))

# 別名は変更前の値を観測する (値意味論の保存)
m = Matrix[Float[32], 2, 2]([[1.0, 2.0], [3.0, 4.0]])
alias = m
j = 0
while j < 2:
    m = m + Matrix[Float[32], 2, 2].ones()
    j = j + 1
print(from_prim(m))
print(from_prim(alias))

# 複数 carried + ネストループ + continue
u = Vector[Float[64], 3].zeros()
v = Vector[Float[64], 3].ones()
for k in range(4):
    if k == 1:
        continue
    u = u + v
    v = v * 2.0
print(from_prim(u))
print(from_prim(v))

n = Matrix[Float[32], 2, 2].zeros()
for p in range(2):
    for q in range(3):
        n = n + Matrix[Float[32], 2, 2].ones()
print(from_prim(n[0, 0]))
