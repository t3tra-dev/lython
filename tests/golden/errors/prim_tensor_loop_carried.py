from lyrt import from_prim
from lyrt.prim import Float, Matrix

a = Matrix[Float[32], 2, 2]([[1.0, 2.0], [3.0, 4.0]])
da = Matrix[Float[32], 2, 2]([[0.5, 0.5], [0.5, 0.5]])
i = 0
while i < 3:
    a = a + da
    i = i + 1
print(from_prim(a))
