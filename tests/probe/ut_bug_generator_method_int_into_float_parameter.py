# WHAT: An int passed to a float parameter of a generator METHOD is refused
#   ("call arguments do not match the Callable contract"). The same call to a
#   generator FUNCTION compiles and prints CPython's answer, and so does a
#   plain method: those are specialized at the argument's rung, and a
#   generator method is called through its one symbol.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-21, testing ut_bug_generator_method_fixed_tuple_argument's
#   neighbours (present before that fix).
# CLASSIFICATION @ lang-gaps 1af75821: 1 拒否
# CPython 3.14 expects: [1.5]
from collections.abc import Iterator


class G:
    def half(self, v: float) -> Iterator[float]:
        yield v / 2


print(list(G().half(3)))
