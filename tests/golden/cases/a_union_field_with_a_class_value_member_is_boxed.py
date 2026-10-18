# WHAT: a field whose union has a class-value member (`int | type[A]`) holds
# either member, reads each back, and the instance goes into a container; the
# same union is a list element, a dict value and a return.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the union is stored as one
# box whose class word names the member, and the class value's box names
# `type` -- the read has to rebuild the class value from that box, and only
# printing what it rebuilt shows which class it names. Until class values
# carried their type object the field kept every member's lanes inline and
# `print([U()])` was refused as "expands to 3 physical values".
class A:
    pass


class B(A):
    pass


class U:
    def __init__(self) -> None:
        self.a: "int | type[A]" = 1

    def show(self) -> str:
        v = self.a
        if isinstance(v, int):
            return "int " + str(v)
        return "class " + v.__name__

    def __repr__(self) -> str:
        return "U"


u = U()
print(u.show())
u.a = B
print(u.show())
us = [U(), U()]
us[1].a = A
print(us, [x.show() for x in us])
xs: list[int | type[A]] = [1, B, 2, A]
for x in xs:
    print(x)
d: dict[str, int | type[A]] = {"k": B}
print(d["k"], d)


def pick(n: int) -> "int | type[A]":
    if n != 0:
        return B
    return 7


print(pick(0), pick(1), [pick(0), pick(1)])
