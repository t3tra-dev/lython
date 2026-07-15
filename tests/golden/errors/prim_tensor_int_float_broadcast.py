from lyrt.prim import Int, Matrix

# float スカラーは整数 tensor へ broadcast できない: 要素 dtype を暗黙に
# 変える演算は width 明示のプリミティブでは提供しない

i = Matrix[Int[32], 2, 2].ones()
c = i * 2.5
