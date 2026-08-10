# Why execution: the point is the VALUE the reused name still holds. A message
# handed to an exception constructor is transferred by `__init__`'s contract,
# so before the caller's retain was restored this program did not compile at
# all; if the retain were ever dropped again it would compile and print freed
# memory, which only running can tell from the correct output.
#
# Three shapes in one subject, because the fix is one count rather than three
# arms: read after one consume, two consumes then a read, and a consume with
# nothing after it (the `raise` shape, which must keep taking no retain).


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


def main() -> None:
    one_consume_then_read()
    two_consumes_then_read()
    consume_with_nothing_after()


main()
