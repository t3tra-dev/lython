class Key:
    def __init__(self, v: int) -> None:
        self.v = v

    def __hash__(self) -> int:
        return self.v * 7 + 1


k = Key(6)
print(hash(k))
print(hash(Key(0)))
print(abs(3 + 4j))
print(abs(-3 - 4j))
print(abs(-5))
print(abs(2.5))
print(abs(-2.5))
print(abs(-10**25))
print(hash(42))
