class Counter:
    def __init__(self, value: int) -> None:
        self.value = value

    def increment(self, amount: int) -> None:
        self.value += amount


c = Counter(42)
print(c.value)

c.increment(8)
print(c.value)
