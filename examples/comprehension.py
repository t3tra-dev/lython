def main() -> None:
    flat = [v for row in [[1, 2], [3, 4]] for v in row]
    print(flat)
    pairs = [x * 10 + y for x in range(3) for y in range(2) if x != y]
    print(pairs)
    grid = [[y for y in range(x + 1)] for x in range(3)]
    print(grid)
    xs: list[int] = []
    for row in [[5, 6], [7]]:
        for v in row:
            xs.append(v)
    print(xs)


main()
