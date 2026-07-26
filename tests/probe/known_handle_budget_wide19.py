# probe: REPORTED loud (budget 5): a JSONValue-shaped class with many handles
# axes: width=wNcls(19) op=construct flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 0  0

class JSONValue:
    def __init__(self) -> None:
        self.kind: int = 0
        self.b: bool = False
        self.i: int = 0
        self.d: float = 0.0
        self.s: str = ""
        self.items: list[int] = []
        self.keys: list[str] = []
        self.vals: list[int] = []


v = JSONValue()
print(v.kind, v.s, len(v.items))
