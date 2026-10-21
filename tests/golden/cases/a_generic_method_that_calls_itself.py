# WHAT: generic methods that call themselves -- counting the rest of a list,
# and appending each element to an accumulator the call passes on -- solved
# for int and for str, and one given an empty list literal for the
# accumulator.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the recursion was refused
# ("calls itself") while each solution's body was copied into its call; now it
# is a call of that solution's body, and whether each call reached the body of
# its own solution is seen only in the values it returns. The empty literal
# became a `list[T]` value that no solution accepted.
class Walker:
    def count[T](self, xs: list[T], i: int) -> int:
        if i == len(xs):
            return 0
        return 1 + self.count(xs, i + 1)

    def walk[T](self, xs: list[T], i: int, out: list[T]) -> list[T]:
        if i < len(xs):
            out.append(xs[i])
            return self.walk(xs, i + 1, out)
        return out

    def put[T](self, xs: list[T], out: list[T]) -> list[T]:
        for x in xs:
            out.append(x)
        return out


w = Walker()
print(w.count([1, 2], 0), w.count(["a", "b", "c"], 0), Walker().count([], 0))
ints: list[int] = []
strs: list[str] = []
print(w.walk([3, 4], 0, ints), w.walk(["x", "y"], 1, strs), ints, strs)
print(w.put([5, 6], []), w.put(["z"], []), len(w.put([1.5], [])))
