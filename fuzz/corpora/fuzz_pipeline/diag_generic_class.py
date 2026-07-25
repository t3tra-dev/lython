class Store[T]:
    def __init__(self) -> None:
        self.items: dict[int, T] = {}


class Row[*Ts]:
    def __init__(self, width: int) -> None:
        self.width = width


class Shaped[**P]:
    def __init__(self, tag: str) -> None:
        self.tag = tag


ungrounded = Store()
pack: Row[int, str] = Row(1)
spec: Shaped[int] = Shaped("t")
print(Store.items)
print(Store[int, str])
