def divmod10(n: int) -> tuple[int, int]:
    return (n // 10, n % 10)


q, r = divmod10(42)
print(q, r)

pair: tuple[float, str] = (2.5, "x")
print(pair)
