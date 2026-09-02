# What: a bool is a member of the returned union like any other member, so the
# caller gets a flag back where the function returned one and a number where it
# returned the other. Printing them is the decode that separates the two: a
# bool widened to an int would print 0 here, not False, and `isinstance` would
# then answer the same for both.
def classify(n: int):
    if n < 0:
        return False
    return n * 2


for value in (-2, 0, 3):
    print(classify(value))

print(isinstance(classify(-1), bool), isinstance(classify(1), bool))


def pick(flag: bool):
    return "many" if flag else False


print(pick(True), pick(False))


def three(n: int):
    if n == 0:
        return False
    if n == 1:
        return "one"
    return n * 10


for i in range(4):
    print(three(i))


def name(n: int) -> str:
    got = classify(n)
    if isinstance(got, bool):
        return "no"
    return "yes"


print(name(-5), name(5))
