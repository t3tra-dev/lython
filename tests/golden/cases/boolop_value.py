def pick(v: str | None) -> str:
    return v or "default"
print(pick(None))
print(pick(""))
print(pick("chosen"))
def first(a: str, b: str) -> str:
    return a or b
print(first("", "second"))
print(first("head", "tail"))
def both(a: str, b: str) -> str:
    return a and b
print(both("", "x"))
print(both("y", "z"))
def guard(v: list[int] | None) -> list[int]:
    return v or [0]
print(guard(None))
print(guard([4, 5]))
