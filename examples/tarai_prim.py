from lyrt import from_prim, native
from lyrt.prim import Int

p1 = Int[8](1)


@native(gc="none")
def tarai(x: Int[8], y: Int[8], z: Int[8]) -> Int[8]:
    if x > y:
        return tarai(tarai(x - p1, y, z), tarai(y - p1, z, x), tarai(z - p1, x, y))
    else:
        return y


ans = tarai(Int[8](14), Int[8](7), Int[8](0))
print(from_prim(ans))
