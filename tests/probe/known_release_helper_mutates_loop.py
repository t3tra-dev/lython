# probe: REPORTED loud (B7): a mutated list handed to a helper inside a loop
# axes: op=pass-to-function flow=for
# CLASSIFICATION @ kernel/4b fa71a3c: 3 loud 拒否 (診断)
#   borrowed entry argument 0 of @put is released or transferred without a prior retain
# CPython 3.14 expects: 3

def put(xs: list[int], v: int) -> None:
    xs.append(v)


xs: list[int] = []
for i in range(3):
    put(xs, i)
print(len(xs))
