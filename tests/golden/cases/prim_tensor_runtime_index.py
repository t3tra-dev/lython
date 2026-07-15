from lyrt import from_prim
from lyrt.prim import Float, Int, Matrix

# 添字はリテラルでなくてもよい: ランタイム値は静的 shape を境界として
# 実行時に範囲検査され、範囲外は trap する (静的に分かる範囲外は従来どおり
# emit error)

m = Matrix[Float[32], 2, 3]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Python int のランタイム添字
i = 1
print(from_prim(m[i, 0]))

# prim Int の添字
j = Int[32](2)
print(from_prim(m[1, j]))

# 負のランタイム添字は Python 同様に後ろから数える
k = -1
print(from_prim(m[0, k]))

# ループ変数で全要素を舐めて合計する (ベンチマークの検算パターン)
s = 0.0
for r in range(2):
    for c in range(3):
        s = s + from_prim(m[r, c])
print(s)

# setitem も同じ添字経路に乗る
x = Matrix[Float[32], 2, 2].zeros()
p = 1
x[p, p] = 9.0
print(from_prim(x[1, 1]))
