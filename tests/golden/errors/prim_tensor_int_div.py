from lyrt import from_prim
from lyrt.prim import Int, Matrix

# 整数要素の / は拒否される: Python の / は int でも float を返すので、
# dtype を暗黙に変える演算は width 明示のプリミティブでは提供しない

a = Matrix[Int[32], 2, 2].ones()
b = Matrix[Int[32], 2, 2].full(2)
print(from_prim(a / b))
