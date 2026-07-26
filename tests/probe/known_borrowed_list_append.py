# probe: REPORTED loud: list append on a borrowed (parameter) list
# axes: acquire=param width=w3list op=append flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 3 loud 拒否 (診断)
#   borrowed entry argument 0 of @put is released or transferred without a prior retain
# CPython 3.14 expects: 2 2

def put(xs: list[int]) -> None:
    xs.append(2)


xs: list[int] = [1]
put(xs)
print(len(xs), xs[1])
