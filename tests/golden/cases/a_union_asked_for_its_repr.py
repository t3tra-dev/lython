# What: repr of a union renders the live member the repr way, which is not the
# same string str gives it -- a str member comes back quoted and a non-ASCII one
# comes back escaped under !a. Printing both spellings of the same value side by
# side is the decode: if the union fell through to str, the quotes would be
# missing and the two columns would agree.
def pick(n: int):
    if n < 0:
        return "a b"
    return n * 3


for value in (-1, 2):
    got = pick(value)
    print(repr(got), str(got), f"{got!r}", f"{got}", "%r" % (got,))


def maybe(n: int):
    if n < 0:
        return None
    return n


print(repr(maybe(-1)), repr(maybe(4)))

table = {"a": 1}
print(repr(table.get("a")), repr(table.get("b")))


def wide(n: int):
    if n < 0:
        return "né"
    return n


print(ascii(wide(-1)), ascii(wide(1)), f"{wide(-1)!a}")


class Tag:
    def __repr__(self) -> str:
        return "Tag()"


def tagged(n: int):
    if n < 0:
        return Tag()
    return n


print(repr(tagged(-1)), repr(tagged(1)))
print([repr(pick(-1)), repr(pick(1))])
