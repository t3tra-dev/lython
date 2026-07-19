class Key:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __hash__(self) -> int:
        return hash(self.tag)

    def __eq__(self, other: Key) -> bool:
        return self.tag == other.tag

    def __repr__(self) -> str:
        return "Key(" + repr(self.tag) + ")"


d: dict[Key, int] = {}
d[Key("a")] = 1
d[Key("b")] = 2
print(d[Key("a")])
print(d[Key("b")])
print(len(d))
d[Key("a")] = 10
print(d[Key("a")], len(d))
print(Key("b") in d, Key("zz") in d)
print(d[Key("missing")])
