from lyrt import from_prim
from lyrt.prim import Float, Matrix

# break を含むループでの shaped primitive 再代入は拒否される: break エッジは
# carried バッファをループ外へ持ち出し、後続イテレーションの書き戻しと
# 区別できない

c = Matrix[Float[32], 2, 2].zeros()
j = 0
while j < 10:
    c = c + Matrix[Float[32], 2, 2].ones()
    if j >= 1:
        break
    j = j + 1
print(from_prim(c))
