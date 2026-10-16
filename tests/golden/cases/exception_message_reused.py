# Why execution: the point is the VALUE the reused name still holds. A message
# handed to an exception constructor is transferred by `__init__`'s contract,
# so before the caller's retain was restored this program did not compile at
# all; if the retain were ever dropped again it would compile and print freed
# memory, which only running can tell from the correct output.
#
# Three shapes in one subject, because the fix is one count rather than three
# arms: read after one consume, two consumes then a read, and a consume with
# nothing after it (the `raise` shape, which must keep taking no retain).
#
# And three where the read is NOT in the consume's block, which the count
# could not see and the program was refused for ("used after release", or the
# unwind path "still owned"): a read in a branch, a read reached only through
# the handler of a raise that took the name, and two consumes in two blocks --
# two inlined methods that each raise `KeyError(key)` -- before the read.


def one_consume_then_read() -> None:
    msg = "alpha"
    err = ValueError(msg)
    print(str(err))
    print(msg)
    print(len(msg))


def two_consumes_then_read() -> None:
    msg = "beta"
    first = ValueError(msg)
    second = TypeError(msg)
    print(str(first))
    print(str(second))
    print(msg)


def consume_with_nothing_after() -> None:
    msg = "gamma"
    try:
        raise ValueError(msg)
    except ValueError as err:
        print(str(err))


def consume_then_read_in_a_branch(loud: bool) -> None:
    msg = "delta" + str(loud)
    err = ValueError(msg)
    if loud:
        print(msg)
    print(repr(err))
    print(msg)


def raise_then_read_after_the_handler() -> None:
    msg = "epsilon" + str(len("ab"))
    try:
        raise KeyError(msg)
    except KeyError:
        print("caught")
    print(msg + "!")


class Table:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def __getitem__(self, key: str) -> str:
        if key not in self.data:
            raise KeyError(key)
        return self.data[key]

    def __delitem__(self, key: str) -> None:
        if key not in self.data:
            raise KeyError(key)
        del self.data[key]

    def pop(self, key: str) -> str:
        value = self[key]
        del self[key]
        return value


def two_inlined_raises_then_read(table: Table) -> None:
    key = "zeta" + str(len(table.data))
    try:
        del table[key]
    except KeyError as err:
        print("del", err)
    try:
        table.pop(key)
    except KeyError as err:
        print("pop", err)
    print(key + "!")


def main() -> None:
    one_consume_then_read()
    two_consumes_then_read()
    consume_with_nothing_after()
    consume_then_read_in_a_branch(True)
    consume_then_read_in_a_branch(False)
    raise_then_read_after_the_handler()
    two_inlined_raises_then_read(Table())


main()
