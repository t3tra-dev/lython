from lyrt import from_prim
from lyrt.prim import Float, Matrix

print(from_prim(Matrix[Float[32], 2, 2].eye()))
