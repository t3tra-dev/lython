# Why execution: the zero-trip loop raises UnboundLocalError (and NameError at
# module scope) only when the read runs, and only running shows the slot was
# written on the other paths.
#
# A name first bound inside a for / while / try / with lives in a slot that
# records whether it was written. `v = None` there was "unresolved name 'v'":
# a None has no word for the slot's field. So does a class object (`k = A`)
# when nothing derives from A -- its type says which class it is. Both slots
# now hold the written flag alone, and a read rebuilds the value.
class Ctx:
    def __enter__(self) -> "Ctx":
        return self

    def __exit__(self, a: object, b: object, c: object) -> None:
        pass


def in_for(n: int) -> str:
    for _ in range(n):
        v = None
    try:
        return "for " + str(v is None) + " " + str(v)
    except UnboundLocalError as e:
        return "for unbound: " + str(e)


def in_while() -> str:
    i = 0
    while i < 2:
        v = None
        i += 1
    return "while " + str(v)


def in_try() -> str:
    try:
        v = None
    finally:
        pass
    return "try " + str(v)


def in_with() -> str:
    with Ctx():
        v = None
    return "with " + repr(v)


print(in_for(1), in_for(0), in_while(), in_try(), in_with())
for _ in range(1):
    top = None
print(top)
for _ in range(0):
    never = None
try:
    print(never)
except NameError as e:
    print("module:", e)


class A:
    def hi(self) -> str:
        return "A"


def pick() -> str:
    for _ in range(1):
        k = A
    return k().hi()


print(pick())
