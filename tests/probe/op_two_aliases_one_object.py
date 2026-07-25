# probe: two locals alias one object; a rebind through one is observed through the other
# axes: acquire=call width=w3list op=alias flow=straight
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 3 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


o = mk()
alias = o
fresh: list[int] = [1, 2, 3]
alias.f = fresh
print(len(o.f), len(alias.f))
