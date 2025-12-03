from lyrt import from_prim
from lyrt.prim import f32, Vector, Matrix, Tensor

v = Vector[f32, 4].zeros()

m = Matrix[f32, 3, 4].zeros()

t = Tensor[f32, 2, 3, 4].zeros()

print(from_prim(v))
print(from_prim(m))
print(from_prim(t))

v2 = Vector[f32, 4]([0.3, -1.2, 4.5, 2.1])

m2 = Matrix[f32, 3, 4]([
    [1.1, -0.7, 3.3, 0.0],
    [2.4, 5.6, -2.2, 1.9],
    [0.5, 4.8, -1.1, 3.7],
])

t2 = Tensor[f32, 2, 3, 4]([
    [
        [0.9, -1.3, 2.2, 4.4],
        [3.1, 0.8, -0.5, 6.6],
        [7.7, -2.4, 1.2, 0.3],
    ],
    [
        [5.5, 2.6, -3.3, 1.4],
        [8.8, -0.9, 4.0, 2.2],
        [6.1, 3.3, -1.7, 9.9],
    ],
])

print(from_prim(v2))
print(from_prim(m2))
print(from_prim(t2))
