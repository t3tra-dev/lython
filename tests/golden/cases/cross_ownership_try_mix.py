class Key:
    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: "Key") -> bool:
        return self.name == other.name

    def __repr__(self) -> str:
        return "Key(" + repr(self.name) + ")"


def run() -> None:
    xs = [1, 2, 3, 4, 5]
    xs[1:4] = [20, 30]
    print(xs)
    del xs[0]
    print(xs)
    try:
        d = {Key("a"): 1, Key("b"): 2}
        print(d[Key("a")])
        fs = frozenset([1, 2, 3, 2, 1])
        print(len(fs))
        print(2 in fs, 9 in fs)
        print(d[Key("missing")])
    except KeyError as e:
        print("caught:", repr(e))
    try:
        raise KeyError("plain")
    except KeyError as e:
        print("caught2:", repr(e))


run()
