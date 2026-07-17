def f(v: int | None) -> int:
    return v or 7
print(f(3))
