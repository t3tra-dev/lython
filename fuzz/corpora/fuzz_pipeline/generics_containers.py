def total(xs: list[int]) -> int:
    acc = 0
    for x in xs:
        acc += x
    return acc


table: dict[str, int] = {"a": 1, "b": 2}
squares = {"k" + str(n): n * n for n in range(4)}
uniq = {v % 3 for v in range(9)}
print(total([1, 2, 3]), table["a"] + table["b"], len(squares), len(uniq))
