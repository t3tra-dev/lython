# The aliases for cases/a_type_alias_read_through_its_module.py, and a class
# alias, which is the class as a value too.
import types

Pair = tuple[int, str]
Info = tuple[type[BaseException], BaseException, types.TracebackType]


class Program:
    def __init__(self, n: int = 1) -> None:
        print("Program", n)


main = Program
