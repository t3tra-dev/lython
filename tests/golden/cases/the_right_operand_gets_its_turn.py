# What: `2 * vec` has to reach the class's `__rmul__`, and `vec * 2` has to
# keep reaching `__mul__`. Running it is what tells the two apart -- both
# answer a Vector, and only the numbers say which method ran.
class Vector:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __mul__(self, k: int) -> "Vector":
        return Vector(self.x * k, self.y * k)

    def __rmul__(self, k: int) -> "Vector":
        return Vector(self.x * k + 100, self.y * k + 100)

    def __radd__(self, k: int) -> int:
        return self.x + self.y + k

    def __rsub__(self, k: float) -> float:
        return k - float(self.x)

    def __repr__(self) -> str:
        return "Vector(" + str(self.x) + ", " + str(self.y) + ")"


v = Vector(1, 2)
print(v * 3, 3 * v)
print(7 + v, 10.0 - v)
print(2 * 3, "ab" * 2, [1] * 2)
