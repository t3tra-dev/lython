# Why execution: the ORDER of the prints is the whole assertion -- enter, body,
# exit, and for nested managers the exits in reverse. None of these compiled
# ("runtime manifest has no Ctx.__enter__ method"), so nothing but running
# them shows the sequence.
#
# The second half is the reason the first half did not work on its own: a
# `return` inside an inlined method body used to run the `with` cleanups of
# the CALL SITE. For __exit__ that closed a cycle into itself; for any other
# method it tore the block down early, which only the order reveals.


class Ctx:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> str:
        print("enter", self.name)
        return self.name

    def __exit__(self, a: object, b: object, c: object) -> bool:
        print("exit", self.name)
        return False


class Boom:
    def __enter__(self) -> str:
        print("enter boom")
        raise ValueError("v")

    def __exit__(self, a: object, b: object, c: object) -> bool:
        print("exit boom")
        return False


class Plain:
    def value(self) -> int:
        return 7


def simple() -> None:
    with Ctx("a") as v:
        print("body", v)


def nested() -> None:
    with Ctx("a") as a:
        with Ctx("b") as b:
            print("body", a, b)


def other_method_inside() -> None:
    with Ctx("a"):
        print("value", Plain().value())
        print("still inside")


def loop_inside() -> None:
    with Ctx("a"):
        for i in range(2):
            print("i", Plain().value() + i)


def raises_inside() -> None:
    try:
        with Ctx("a"):
            raise ValueError("v")
    except ValueError as e:
        print("caught", e)


def early_return() -> int:
    with Ctx("a"):
        return 1
    return 0


def break_out() -> None:
    for i in range(3):
        with Ctx("a"):
            if i == 1:
                break
            print("body", i)


def continue_out() -> None:
    for i in range(3):
        with Ctx("a"):
            if i == 1:
                continue
            print("body", i)


def handled_inside_then_return() -> int:
    with Ctx("a"):
        try:
            raise ValueError("v")
        except ValueError:
            return 7
    return 0


def two_items() -> None:
    with Ctx("a") as a, Ctx("b") as b:
        print("body", a, b)


def second_enter_raises() -> None:
    # CPython enters A, opens A's try, THEN enters B -- so A's __exit__ runs.
    try:
        with Ctx("a"), Boom():
            print("unreachable")
    except ValueError as e:
        print("caught", e)


def main() -> None:
    simple()
    nested()
    other_method_inside()
    loop_inside()
    raises_inside()
    print(early_return())
    break_out()
    continue_out()
    print(handled_inside_then_return())
    two_items()
    second_enter_raises()


main()
