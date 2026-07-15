from lyrt import from_prim
from lyrt.prim import Float, Matrix

# 512^3 以上の行列積は M 方向チャンクの fork-join で複数コアに分配される
# (LYTHON_NUM_THREADS で制御)。結果はスレッド数に依存しないこと

A = Matrix[Float[32], 512, 512].full(0.5)
B = Matrix[Float[32], 512, 512].ones()
C = A @ B
print(from_prim(C[0, 0]))
print(from_prim(C[511, 511]))

# loop-carried との組合せ (毎反復ディスパッチ)
x = Matrix[Float[32], 512, 512].full(0.001953125)
for k in range(3):
    x = A @ x
print(from_prim(x[256, 256]))
