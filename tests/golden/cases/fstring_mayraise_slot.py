from dataclasses import dataclass


@dataclass
class P:
    x: int
    y: int


p = P(1, 2)
q = P(1, 2)
r = P(3, 4)
print(f"eq={p == q} done")
print(f"one={p == q} two={p == r} three={q == r}")
print(f"obj={P(5, 6)} end")
print(f"lead {p == q} mid {P(7, 8)} tail")
