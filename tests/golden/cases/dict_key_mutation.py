class K:
    def __init__(self, v: int) -> None:
        self.v = v

    def __hash__(self) -> int:
        return self.v * 7

    def __eq__(self, other: "K") -> bool:
        return self.v == other.v

    def __repr__(self) -> str:
        return "K(" + repr(self.v) + ")"


k = K(1)
d = {}
d[k] = "a"
k.v = 2
print(K(1) in d)
print(K(2) in d)
print(k in d)
print(len(d))
print(d[K(1)] if K(1) in d else "gone")
