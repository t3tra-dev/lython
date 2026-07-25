# probe: augmented assignment (+=) to a list field of a call-obtained object
# axes: acquire=call width=w3list op=augassign flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 3 loud 拒否 (診断)
#   attribute value '!py.contract<"builtins.list">' is not assignable to field '!py.contract<"builtins.list", [!py.contract<"builtins.int">]>'
# CPython 3.14 expects: 3 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


o = mk()
o.f += [2, 3]
print(len(o.f), o.f[2])
