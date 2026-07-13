# `==` / `!=` on `Optional[T]` values. The comparison dispatches on the union's
# active member: a `None` member is unequal to any present value (and equal only
# to another `None`), while a present member re-enters the ordinary scalar
# comparison. Works against a concrete operand and between two Optionals, for
# both heap (`str`) and unboxed (`int`) members.


def eq_const(x: int | None) -> str:
    r: str = ""
    if x == 5:
        r = r + "eq "
    if x != 5:
        r = r + "ne "
    if 5 == x:
        r = r + "rev "
    return r


def eq_pair(a: int | None, b: int | None) -> int:
    if a == b:
        return 1
    return 0


def label(n: int) -> str | None:
    if n == 0:
        return None
    return "tag" + str(n)


def same_label(a: int, b: int) -> int:
    if label(a) == label(b):
        return 1
    return 0


print(eq_const(5))
print(eq_const(None))
print(eq_const(3))

print(eq_pair(5, 5))
print(eq_pair(5, 3))
print(eq_pair(None, None))
print(eq_pair(5, None))

print(same_label(1, 1))
print(same_label(1, 2))
print(same_label(0, 0))
print(same_label(1, 0))
