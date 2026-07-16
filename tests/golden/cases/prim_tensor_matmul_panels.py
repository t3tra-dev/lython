from lyrt import from_prim
from lyrt.prim import Float, Matrix

# B が閾値以上 (2000x1600x4B = 12.2MB) なので既定で panel-pack 経路に乗り、
# M=2000 は 8x250 にチャンクされ 250 % 32 != 0 の尾部パネルを踏む。
# 値はチャンク相対パネルの全域 (先頭/尾部/チャンク境界) を突く

A = Matrix[Float[32], 2000, 2000].full(0.5)
B = Matrix[Float[32], 2000, 1600].ones()
C = A @ B
print(from_prim(C[0, 0]))
print(from_prim(C[249, 1599]))
print(from_prim(C[250, 0]))
print(from_prim(C[1999, 1599]))

# loop-carried: A のパネルはループ外で 1 回だけ作られる
# 2^-10 で 2 反復: 全ての部分和が f32 の 2^24 ulp に収まり exact なので、
# 総和順序 (カーネル実装や経路) に依らず同じ値になる
x = Matrix[Float[32], 2000, 1600].full(0.0009765625)
for i in range(2):
    x = A @ x
print(from_prim(x[1975, 800]))
