# What: a `__del__` that stores its object somewhere that outlives it -- which
# CPython allows, and which resurrects the object -- stops the program with a
# message that says so, rather than freeing an object something still holds.
# WHY THIS IS RUN: whether the object came back is known only after the
# finalizer returned, at run time. The stop is an assertion's: its message is
# on stdout and the process ends on SIGABRT.
class A:
    def __del__(self) -> None:
        keep.append(self)


keep: list[A] = []


def f() -> None:
    a = A()


f()
print("after")
