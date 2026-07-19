class Key:
    def __init__(self, v: int) -> None:
        self.v = v

    def __eq__(self, other: "Key") -> bool:
        return self.v == other.v

    def clone(self) -> "Key":
        return Key(self.v)


k = Key(3)
m: "Key" = k.clone()
print(m.v)
print(k == m)
print(k is m)
