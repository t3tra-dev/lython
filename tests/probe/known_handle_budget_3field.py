# probe: REPORTED loud (budget 5): a 3-field class used as a list element
# axes: width=wNcls op=store-into-container flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 1 2 3

class Three:
    def __init__(self, a: int, b: int, c: int) -> None:
        self.a: int = a
        self.b: int = b
        self.c: int = c


xs: list[Three] = [Three(1, 2, 3)]
print(xs[0].a, xs[0].b, xs[0].c)
