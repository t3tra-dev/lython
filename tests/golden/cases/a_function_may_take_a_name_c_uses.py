# WHAT: the program's own functions may be called what C and the runtime call
# theirs -- `abort`, `free`, `malloc`, `puts`, `write`, `fmod`, `strtod`,
# `memcpy`, `main`, `exit` at the top level, a method called `abort`, a nested
# `free` -- and both sides keep their own: the program's calls reach its
# functions, and the runtime's float `%`, `float(str)` and output still reach
# C's.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which function each call
# reaches is what is printed. Each of these names was refused while the
# runtime was merged in ("redefinition of symbol named 'abort'", "redefinition
# of reserved function 'free'", a call into float.mlir that no longer
# matched), because a Python function was emitted under its bare name.


def abort(reason: str) -> str:
    return "abort:" + reason


def free(n: int) -> int:
    return n - 1


def malloc(n: int) -> list[int]:
    return [0] * n


def puts(s: str) -> int:
    return len(s)


def write(fd: int, text: str) -> str:
    return str(fd) + ">" + text


def fmod(a: float, b: float) -> float:
    return -1.0


def strtod(s: str) -> float:
    return -2.0


def memcpy(dst: list[int], src: list[int]) -> list[int]:
    return dst + src


def main() -> int:
    return 7


def exit(code: int) -> str:
    return "not exiting " + str(code)


class Process:
    def abort(self) -> str:
        return "Process.abort"


def scope() -> int:
    def free(n: int) -> int:
        return n * 100
    return free(3)


x = 7.5
y = 2.0
print(abort("x"), free(5), malloc(2), puts("abc"), write(1, "hi"))
print(fmod(1.0, 2.0), strtod("3"), memcpy([1], [2]), main(), exit(3))
print(Process().abort(), scope())
print(x % y, float("2.5"), -7.5 % 2.0)
