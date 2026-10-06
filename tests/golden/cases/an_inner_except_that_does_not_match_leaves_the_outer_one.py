# WHAT: an exception that the inner `except` of two nested `try`s in ONE frame
# does not name is caught by the outer `except` of that same frame -- for a
# call, a raise, a division, through an inlined method, three levels deep,
# and past a `finally` -- and one the inner `except` names is still caught
# there.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: whether the outer handler
# runs is decided by the unwinder reading the landing pad's clause list at run
# time; the IR only shows the list, and a list that leaves the outer class out
# looks like any other.


def boom(n: int) -> int:
    if n > 0:
        raise ValueError("boom")
    return n


class Box:
    def divide(self, n: int) -> int:
        return 7 // n

    def through(self, n: int) -> str:
        try:
            try:
                return str(self.divide(n))
            except KeyError:
                return "inner"
        except ZeroDivisionError as e:
            return f"outer {e}"


def call(n: int) -> str:
    try:
        try:
            boom(n)
        except KeyError:
            return "inner"
    except ValueError as e:
        return f"outer {e}"
    return "none"


def raised(n: int) -> str:
    try:
        try:
            if n != 0:
                raise TypeError("raised")
        except (KeyError, IndexError):
            return "inner"
    except TypeError as e:
        return f"outer {e}"
    return "none"


def divided(n: int) -> str:
    try:
        try:
            return str(1 // n)
        except KeyError:
            return "inner"
    except ZeroDivisionError as e:
        return f"outer {e}"


def three(n: int) -> str:
    try:
        try:
            try:
                boom(n)
            except KeyError:
                return "first"
        except IndexError:
            return "second"
    except ValueError as e:
        return f"third {e}"
    return "none"


def past_finally(n: int) -> str:
    try:
        try:
            boom(n)
        finally:
            print("finally runs")
    except ValueError as e:
        return f"outer {e}"
    return "none"


def named_inside(n: int) -> str:
    try:
        try:
            {}["k"] if n != 0 else 0
        except KeyError:
            return "inner"
    except Exception:
        return "outer"
    return "none"


print(call(1), call(0))
print(raised(1), raised(0))
print(divided(0), divided(1))
print(Box().through(0), Box().through(2))
print(three(1), three(0))
print(past_finally(1))
print(named_inside(1), named_inside(0))
