# probe: borrowed list extend (grow)
# axes: acquire=param width=w3list op=extend flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   borrowed entry argument 0 of @put is released or transferred without a prior retain
# CPython 3.14 expects: 3

def put(xs: list[int]) -> None:
    xs.extend([2, 3])


xs: list[int] = [1]
put(xs)
print(len(xs))
