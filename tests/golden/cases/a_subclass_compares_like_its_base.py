# What: a base declares the comparison and a subclass inherits it, so a list
# holding both has to sort and to answer `in`. The decode is that each answer
# is asked of a MIXED pair -- an all-P list and an all-Q list would both agree
# with a compiler that only ever compares like with like.
class Point:
    def __init__(self, n: int) -> None:
        self.n = n

    def __lt__(self, other: "Point") -> bool:
        return self.n < other.n

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and other.n == self.n

    def __repr__(self) -> str:
        return "Point(" + str(self.n) + ")"


class Marked(Point):
    pass


mixed: "list[Point]" = [Marked(2), Point(1), Marked(3)]
print([p.n for p in sorted(mixed)])
print(Point(2) in mixed, Point(9) in mixed, Marked(1) in mixed)

ordered = list(mixed)
ordered.sort()
print(ordered)

print(Point(1) == Marked(1), Marked(1) == Point(1), Point(1) == Point(2))
print(Point(1) < Marked(2), Marked(2) < Point(1))
print(max(mixed).n, min(mixed).n)


# The same list with only one class still behaves, and so do the builtins that
# reach the same comparison.
same: "list[Point]" = [Point(2), Point(1)]
print([p.n for p in sorted(same)])
print(sorted([3, 1, 2]), sorted(["c", "a"]), sorted([1.5, 0.5]))

erased: "list[object]" = [1, "a", 2.5, True]
print(1 in erased, "a" in erased, 2.5 in erased, 9 in erased)
