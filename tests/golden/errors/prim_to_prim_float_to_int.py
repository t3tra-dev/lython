from lyrt import to_prim
from lyrt.prim import Float, Int, Matrix

# float -> int の tensor キャストは拒否される: linalg のキャスト本体は
# 要素ごとに trap できず、範囲外の値が黙って壊れる

a = Matrix[Float[32], 2, 2].ones()
b = to_prim(a, Matrix[Int[32], 2, 2])
