from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int

    def __format__(self, spec: str) -> str:
        if spec == "":
            return "(" + str(self.x) + ", " + str(self.y) + ")"
        return format(self.x, spec) + ";" + format(self.y, spec)


p = Point(3, 4)
print(f"{p!r}")
print(f"{p}")
print(f"{p:>5d}")
print(f"repr={p!r} plain={p} spec={p:03d}")
q = Point(3, 4)
w = Point(0, 0)
eq1 = p == q
eq2 = p == w
print(f"{eq1} {eq2}")
