# What: a helper defined inside a function has to be able to call itself and
# its siblings, and only running the recursion shows that the call reached the
# same closure -- each of these reads a different captured value.
def sum_to(n: int) -> int:
    def rec(k: int) -> int:
        if k <= 0:
            return 0
        return k + rec(k - 1)

    return rec(n)


print(sum_to(4), sum_to(0))


def total(values: "list[int]") -> int:
    def walk(index: int) -> int:
        if index >= len(values):
            return 0
        return values[index] + walk(index + 1)

    return walk(0)


print(total([1, 2, 3]), total([]))


def chained(n: int) -> int:
    def helper(k: int) -> int:
        return k + 1

    def rec(k: int) -> int:
        if k <= 0:
            return 0
        return helper(k) + rec(k - 1)

    return rec(n)


print(chained(3))


def depth(text: str) -> int:
    def count(index: int, seen: int) -> int:
        if index >= len(text):
            return seen
        return count(index + 1, seen + 1)

    return count(0, 0)


print(depth("abcd"))
