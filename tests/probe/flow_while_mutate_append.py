# probe: in-place field mutation inside a while loop (append)
# axes: acquire=call width=w3list/w1dict op=append flow=while
# CLASSIFICATION: 3 loud 拒否 (診断)
#   list.append on a field or borrowed list is not supported inside a branch or loop body; bind the list to a local variable and mutate the local instead
# CPython 3.14 expects: 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
k = 0
while k < 3:
    o.f.append(k)
    k = k + 1
print(len(o.f))
