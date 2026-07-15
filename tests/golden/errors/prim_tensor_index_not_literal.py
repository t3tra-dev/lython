from lyrt import from_prim
from lyrt.prim import Float, Matrix

m = Matrix[Float[32], 2, 3]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
i = 0
print(from_prim(m[i, 0]))
