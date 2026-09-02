# What: an operator applied to a value whose static type is a union. CPython
# does not know which member is live either -- it asks the object at run time,
# and both members answer -- so the compiler has to ask the tag. Every result
# here is decoded rather than only printed: the numeric ones are compared and
# summed, and the str one is measured, so a wrong arm would not merely render
# differently.
def scaled(n: int):
    if n < 0:
        return 1.5
    return n


print(scaled(-1) + 1, scaled(3) + 1)
print(scaled(-1) * 2, scaled(3) * 2)
print(scaled(-1) / 2, scaled(4) / 2)
print(scaled(-1) - 0.5, scaled(3) - 1)
print(-scaled(-1), -scaled(3))
print(scaled(-1) < 2, scaled(3) < 2)


def flagged(n: int):
    if n < 0:
        return False
    return n


print(flagged(-1) + 1, flagged(3) + 1)
print(flagged(-1) * 5, flagged(3) * 5)
print(-flagged(-1), -flagged(3))


def repeated(n: int):
    if n < 0:
        return "ab"
    return n


doubled = repeated(-1) * 2
print(doubled, len(doubled) if isinstance(doubled, str) else doubled)
print(repeated(3) * 2)


# The sum of a whole run, so every arm has to have produced a number.
def run_total(count: int) -> float:
    total = 0.0
    for i in range(count):
        total += scaled(i - 2) + 1
    return total


print(run_total(5))


# The union on the RIGHT of the operator, which is the accumulator shape, and
# one on BOTH sides -- the left's tag chooses a member and the right's tag is
# met with a concrete receiver.
print(1 + scaled(-1), 1 + scaled(3), 2 * flagged(-1))
print(scaled(-1) + scaled(3), scaled(3) + scaled(-1))
print(scaled(-1) + scaled(-1), scaled(3) + scaled(3))
print(flagged(-1) + flagged(3), flagged(3) * flagged(3))


# The same tag dispatch outside the operators: `len` asks each member too, and
# a signature that accepts either is ordinary Python.
def size(x: "list[int] | str") -> int:
    return len(x)


print(size([1, 2, 3]), size("ab"), size([]) + size(""))


def measured(n: int):
    if n < 0:
        return "abc"
    return [1, 2]


print(len(measured(-1)), len(measured(1)), len(measured(-1)) + len(measured(1)))
