# probe: REPORTED loud: dict setitem on a borrowed (parameter) dict
# axes: acquire=param width=w1dict op=setitem flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 3 loud 拒否 (診断)
#   borrowed entry argument 0 of @put is released or transferred without a prior retain
# CPython 3.14 expects: 2 2

def put(d: dict[str, int]) -> None:
    d["b"] = 2


d: dict[str, int] = {"a": 1}
put(d)
print(len(d), d["b"])
