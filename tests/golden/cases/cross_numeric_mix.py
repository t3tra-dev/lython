from typing import Optional

big = 10 ** 30
print(big)
print(big + 1)
print(f"{big:d}")

z = 1 + 2j
w = z * z
print(z)
print(w)
print(f"{z}")


def pick(v: Optional[int]) -> str:
    if v:
        return "truthy"
    return "falsy"


print(pick(None))
print(pick(0))
print(pick(7))
print(pick(big))

opt_f: Optional[float] = 0.0
if opt_f:
    print("bad")
else:
    print("zero float is falsy")
