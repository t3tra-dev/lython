# What: an exception `__del__` raises does not propagate. It is reported on
# stderr the way sys.unraisablehook reports it -- the deallocator by its
# qualified name, the traceback from `__del__` down, the exception -- and the
# program carries on.
# WHY THIS IS RUN: the report is written while the program runs, between two
# of its lines, and the exit code is the program's.
class Noisy:
    def __del__(self) -> None:
        raise ValueError("in del")


def h() -> None:
    i = Noisy()
    print("in h")


h()
print("after h")
