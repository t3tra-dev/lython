# The exception ENTITY bound by `except ... as e` and assigned to a local
# reaches the code after the try, and is the exception that was handled -- not
# the local's pre-try value. Reading a FIELD out of the entity inside the
# handler already worked; carrying the entity itself out did not, in two
# different ways: an exception-typed local answered its pre-try value (the
# statement's result lanes refuse the type and nothing else carried it), and a
# user exception class WAS carried by a lane, which publishes the borrowed
# current-exception pointer after the handler's discard released it (empty
# str() plainly, SIGSEGV under libgmalloc).
kept: BaseException = ValueError("init")
try:
    raise ValueError("boom")
except ValueError as e:
    kept = e
print(str(kept))
print(str(kept.args[0]))


def in_a_function() -> str:
    # No module global to fall back on here: a function-scope local is SSA, so
    # this is the shape the result lanes own.
    local: BaseException = ValueError("init")
    try:
        raise ValueError("inner")
    except ValueError as e:
        local = e
    return str(local)


print(in_a_function())


class Err(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


def user_exception_class() -> str:
    # A user exception class reaches the lane machinery as an ordinary user
    # class, which is how it came to be carried as a freed pointer.
    local: Err = Err("init")
    try:
        raise Err("user")
    except Err as e:
        local = e
    return str(local)


print(user_exception_class())


def which_handler_ran(kind: int) -> str:
    local: BaseException = ValueError("init")
    try:
        if kind == 1:
            raise ValueError("value")
        raise KeyError("key")
    except ValueError as e:
        local = e
    except KeyError as k:
        local = k
    return str(local)


print(which_handler_ran(1))
print(which_handler_ran(2))


def tuple_handler() -> str:
    local: BaseException = ValueError("init")
    try:
        raise KeyError("tuple")
    except (ValueError, KeyError) as e:
        local = e
    return str(local)


print(tuple_handler())


def only_on_one_branch(take: int) -> str:
    # The handler does not always rebind: the pre-try value has to survive the
    # path that skips the assignment.
    local: BaseException = ValueError("init")
    try:
        raise ValueError("branch")
    except ValueError as e:
        if take > 0:
            local = e
    return str(local)


print(only_on_one_branch(1))
print(only_on_one_branch(0))


def with_else_and_finally() -> str:
    local: BaseException = ValueError("init")
    try:
        raise ValueError("late")
    except ValueError as e:
        local = e
    else:
        print("else")
    finally:
        print("finally")
    return str(local)


print(with_else_and_finally())


def body_then_handler() -> str:
    # Rebound in the body as well, so the body's promotion and the handler's
    # rebind have to agree on one channel.
    local: BaseException = ValueError("init")
    try:
        local = ValueError("body")
        raise ValueError("handler")
    except ValueError as e:
        local = e
    return str(local)


print(body_then_handler())
