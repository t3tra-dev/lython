from lyrt import from_prim
from lyrt.prim import Float, Int, Matrix

# shaped primitive を引数・戻り値に持つ関数は out-param 化された buffer 渡しで
# lowering される: 呼び出し先は借用のみ、確保・解放は呼び出し側が持つ


def mm(
    a: Matrix[Float[32], 8, 8], b: Matrix[Float[32], 8, 8]
) -> Matrix[Float[32], 8, 8]:
    return a @ b


def had(
    a: Matrix[Float[32], 8, 8], b: Matrix[Float[32], 8, 8]
) -> Matrix[Float[32], 8, 8]:
    return a * b


# ネストした呼び出し: 中間結果の buffer も呼び出し側フレームが解放する
def chain(
    a: Matrix[Float[32], 8, 8], b: Matrix[Float[32], 8, 8]
) -> Matrix[Float[32], 8, 8]:
    return mm(mm(a, b), b)


# 複数 return: boundary bufferization は return ごとに out-param へ書き戻す
def pick(
    a: Matrix[Float[32], 8, 8], b: Matrix[Float[32], 8, 8], which: Int[32]
) -> Matrix[Float[32], 8, 8]:
    if which > Int[32](0):
        return a @ b
    return a * b


# 引数をそのまま返す: 呼び出し側 buffer への copy になる
def ident(a: Matrix[Float[32], 8, 8]) -> Matrix[Float[32], 8, 8]:
    return a


x = Matrix[Float[32], 8, 8].ones()
y = Matrix[Float[32], 8, 8].full(2.0)

print(from_prim(mm(x, y)[0, 0]))
print(from_prim(had(x, y)[7, 7]))
print(from_prim(chain(x, y)[3, 3]))
print(from_prim(pick(x, y, Int[32](1))[0, 0]))
print(from_prim(pick(x, y, Int[32](0))[0, 0]))
print(from_prim(ident(y)[4, 4]))

# ループ内で関数越しに matmul した結果を消費する (ベンチマークの基本形)
s = 0.0
for k in range(4):
    c = mm(x, y)
    s = s + from_prim(c[0, 0])
print(s)
