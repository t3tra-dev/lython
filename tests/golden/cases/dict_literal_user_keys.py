class Key:
    def __init__(self, v: int) -> None:
        self.v = v

    def __hash__(self) -> int:
        return self.v * 7

    def __repr__(self) -> str:
        return "Key(" + repr(self.v) + ")"

    def __eq__(self, other: "Key") -> bool:
        return self.v == other.v


d = {Key(1): "one", Key(2): "two", Key(1): "one-again"}
print(len(d))
print(d[Key(1)])
d[Key(3)] = "three"
print(d[Key(3)])
print(Key(2) in d)
print(Key(9) in d)
try:
    print(d[Key(9)])
except KeyError:
    print("missing")
