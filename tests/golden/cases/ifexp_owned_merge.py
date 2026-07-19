from dataclasses import dataclass


@dataclass
class P:
    x: int
    y: int


n = 3
p = P(1, 2) if n > 2 else P(3, 4)
print(p.x)
q = P(5, 6)
r = P(7, 8)
s = q if n > 2 else r
print(s.y)
t = P(9, 10) if n > 5 else P(11, 12)
print(t.x)
print(f"picked={t == P(11, 12)}")
