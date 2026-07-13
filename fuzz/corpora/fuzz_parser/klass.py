class C:
    def __init__(self, n: int) -> None:
        self.n = n

    def __repr__(self) -> str:
        return f"C({self.n})"

print(C(1))
