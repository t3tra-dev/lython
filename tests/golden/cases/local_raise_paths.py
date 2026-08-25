# A raise whose handler is in the same frame does not unwind: the compiler
# names the landing pad, so the raise records the exception and the dispatch
# chain runs by falling into it. Every way that can go wrong needs the program
# to run -- what the arms match, what a `finally` and a `with` still run, what
# an exception raised inside a handler chains to, and where the exception goes
# when the local arms do not want it.


def caught_here() -> str:
    try:
        raise ValueError("a")
    except ValueError as e:
        return "caught " + str(e)
    return "unreachable"


def not_caught_here() -> str:
    try:
        raise KeyError("b")
    except ValueError:
        return "wrong arm"
    return "unreachable"


def finally_still_runs(log: list[str]) -> str:
    try:
        try:
            raise ValueError("c")
        finally:
            log.append("finally")
    except ValueError as e:
        return "after finally " + str(e)
    return "unreachable"


class Guard:
    def __enter__(self) -> "Guard":
        return self

    def __exit__(self, kind: object, value: object, tb: object) -> bool:
        print("exit ran")
        return False


def with_still_exits() -> str:
    try:
        with Guard():
            raise ValueError("d")
    except ValueError as e:
        return "after with " + str(e)
    return "unreachable"


def raised_inside_a_handler() -> str:
    try:
        try:
            raise ValueError("e")
        except ValueError:
            raise KeyError("f")
    except KeyError as e:
        return "chained " + str(e)
    return "unreachable"


def second_arm() -> str:
    try:
        raise KeyError("g")
    except ValueError:
        return "first"
    except KeyError as e:
        return "second " + str(e)
    return "unreachable"


print(caught_here())
try:
    print(not_caught_here())
except KeyError as e:
    print("propagated", str(e))
log: list[str] = []
print(finally_still_runs(log))
print(log)
print(with_still_exits())
print(raised_inside_a_handler())
print(second_arm())
