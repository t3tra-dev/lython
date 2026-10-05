# What: `__del__` runs when the last reference goes, at the point CPython runs
# it: a function's locals at its return, parameters first and then names in the
# order they were first bound; an old value when its name is rebound, after the
# new one is made; a temporary at the end of its statement; an element popped
# from a list; a returned object when the caller's name lets go of it; and the
# main module's names at the end, in the order they were first bound, annotated
# or not and read by a function or not.
# WHY THIS IS RUN: when a finalizer runs is the output, and only running it
# shows the order.
class A:
    def __init__(self, n: int) -> None:
        self.n = n

    def __del__(self) -> None:
        print("del", self.n)


def f() -> None:
    a = A(1)
    b = A(2)
    c = A(3)
    print("end f")


f()


def g(p: A) -> None:
    q = A(5)
    print("end g")


g(A(4))
print("after g")


def k() -> A:
    a = A(6)
    b = A(7)
    return b


r = k()
print("got", r.n)


def rebind() -> None:
    a = A(11)
    a = A(12)
    print("rebound")
    A(13)
    print("temporary")
    xs = [A(14), A(15)]
    xs.pop()
    print("popped")


rebind()


class Holder:
    def __init__(self) -> None:
        self.inner = A(16)

    def drop(self) -> None:
        mine = A(17)
        print("drop", mine.n)


Holder().drop()
print("holder gone")

keep: A = A(8)
plain = A(9)
items: list[A] = [A(18)]


def use() -> None:
    print("use", keep.n)


use()
keep = A(10)
print("module end")
