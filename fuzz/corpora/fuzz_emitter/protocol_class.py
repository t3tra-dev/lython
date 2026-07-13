class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return "Point(" + str(self.x) + ", " + str(self.y) + ")"

    def norm1(self) -> int:
        ax = self.x
        if ax < 0:
            ax = -ax
        ay = self.y
        if ay < 0:
            ay = -ay
        return ax + ay


p = Point(1, -2)
print(p)
print(p.norm1())
