GREETING = "hi"


class Counter:
    count = 0
    step: int = 2
    banner: str = GREETING + "!"

    def __init__(self) -> None:
        self.mine = 100

    def bump(self) -> None:
        Counter.count += Counter.step


class Wide(Counter):
    pass


class Shadow:
    v = 10

    def __init__(self) -> None:
        self.v = 99


print(Counter.count)
print(Counter.banner)
Counter.count += 1
print(Counter.count)
c = Counter()
c.bump()
c.bump()
print(Counter.count)
print(c.count)
print(Wide.count)
Counter.step = 10
c.bump()
print(Counter.count)
s = Shadow()
print(s.v)
print(Shadow.v)
