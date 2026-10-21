# WHAT: A bool-returning function held in a `Callable[[], int]` LOCAL, whose
#   result is then used as an int. At module level the same program prints
#   "2 True"; inside a function the lowering fails: "runtime bundle value 0 for
#   'builtins.int' has type 'i1', but ABI expects 'memref<2xi64>'".
# KIND: バグ (bug) -- loud: well-typed code refused, with an internal message
# FOUND: 2026-10-21, decode-testing the callable refusal for
#   ut_bug_int_function_held_as_float_callable (present before that change).
# CLASSIFICATION @ lang-gaps 5d8e3215: 1 拒否 (内部の診断)
# CPython 3.14 expects: 2 True
from collections.abc import Callable


def flag() -> bool:
    return True


def run() -> None:
    b: Callable[[], int] = flag
    print(b() + 1, b())


run()
