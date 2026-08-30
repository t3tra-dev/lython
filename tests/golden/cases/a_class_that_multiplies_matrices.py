# What: `@` on a class that defines `__matmul__`, and `@=` on one that defines
# `__imatmul__`. Running it is what shows the operator reached the right dunder
# -- a wrong table entry answers with `__mul__`'s result, which for these
# operands is a different number rather than a failure.
class Grid:
    def __init__(self, rows: "list[list[int]]") -> None:
        self.rows = rows

    def __matmul__(self, other: "Grid") -> "Grid":
        size = len(self.rows)
        out: list[list[int]] = []
        for i in range(size):
            row: list[int] = []
            for j in range(size):
                total = 0
                for k in range(size):
                    total += self.rows[i][k] * other.rows[k][j]
                row.append(total)
            out.append(row)
        return Grid(out)

    def __mul__(self, other: "Grid") -> "Grid":
        return Grid([[9, 9], [9, 9]])

    def __imatmul__(self, other: "Grid") -> "Grid":
        self.rows = (self @ other).rows
        return self

    def __repr__(self) -> str:
        return "Grid(" + str(self.rows) + ")"


left = Grid([[1, 2], [3, 4]])
print(left @ Grid([[1, 0], [0, 1]]))
print(left @ left)
left @= Grid([[2, 0], [0, 2]])
print(left)
