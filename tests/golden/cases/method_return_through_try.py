# Why execution: the wrong answer was a VALUE. A method returning out of a try
# yielded None, and the compiler exited 0 -- `print(C().m())` printed None
# where CPython prints 1. The same method as a free function was always right,
# so nothing but running both tells them apart.


class Compute:
    def with_finally(self) -> int:
        try:
            return 1
        finally:
            print("finally ran")

    def with_handler(self) -> int:
        try:
            return 2
        except ValueError:
            return 3

    def returning_str(self) -> str:
        try:
            return "ok"
        finally:
            print("cleanup")

    def raising_then_handled(self) -> int:
        try:
            raise ValueError("x")
        except ValueError:
            return 4


def free_with_finally() -> int:
    try:
        return 5
    finally:
        print("free finally")


def main() -> None:
    machine = Compute()
    print(machine.with_finally())
    print(machine.with_handler())
    print(machine.returning_str())
    print(machine.raising_then_handled())
    print(free_with_finally())


main()
