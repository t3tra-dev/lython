# probe: borrowed set add (grow)
# axes: acquire=param width=w1set op=add flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 3 loud 拒否 (診断)
#   borrowed entry argument 0 of @put is released or transferred without a prior retain
# CPython 3.14 expects: 2

def put(s: set[int]) -> None:
    s.add(2)


s: set[int] = {1}
put(s)
print(len(s))
