from lyrt import from_prim
from lyrt.prim import Float, Matrix, Tensor

# 演算テスト

m4 = Matrix[Float[32], 2, 3](
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)
m5 = Matrix[Float[32], 2, 3](
    [
        [0.5, 1.5, 2.5],
        [3.5, 4.5, 5.5],
    ]
)

print(from_prim(m4 + m5))
print(from_prim(m4 - m5))
print(from_prim(m4 * m5))
print(
    from_prim(
        m4
        @ Matrix[Float[32], 3, 2](
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
    )
)

t4 = Tensor[Float[32], 2, 2](
    [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
)
t5 = Tensor[Float[32], 2, 2](
    [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
)

print(from_prim(t4 + t5))
print(from_prim(t4 - t5))
print(from_prim(t4 * t5))

# 0階テンソルはスカラーと同値
#
# t4 = Tensor[Float[32], ...](3.14)
# assert from_prim(t4) == Float[32](3.14)
#
# print(from_prim(t4))
