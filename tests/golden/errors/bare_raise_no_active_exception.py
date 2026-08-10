# A bare `raise` outside every handler has nothing to re-raise. CPython raises
# RuntimeError; this compiler emitted py.raise.current, whose runtime arm for
# an empty exception slot is a trap -- the program aborted with no output at
# all. The handler that ran earlier does not count: it completed, and CPython
# raises here too.


def main() -> None:
    try:
        raise ValueError("handled")
    except ValueError:
        pass
    raise


main()
