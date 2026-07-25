# probe: REPORTED loud (budget 5): a 3-field class used as a list element
# axes: width=wNcls op=store-into-container flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   a '!py.contract<"Three">' value expands to 10 physical handles, but a payload box carries at most 5; it cannot be stored in a container slot or boxed field yet (reduce the class to fewer or narrower fields)
# CPython 3.14 expects: 1 2 3

class Three:
    def __init__(self, a: int, b: int, c: int) -> None:
        self.a: int = a
        self.b: int = b
        self.c: int = c


xs: list[Three] = [Three(1, 2, 3)]
print(xs[0].a, xs[0].b, xs[0].c)
