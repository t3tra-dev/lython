# Why execution: `True + 1` was refused ("builtins.bool.__add__ is declared by
# the standard-library contract but has no implementation"), and so was every
# other arithmetic, comparison and unary operator over a bool. The values are
# the assertion because the interesting part is WHICH answers stay bools --
# CPython's `True | False` is `True`, not `1` -- and a promotion that is one
# operator too wide is a wrong type, not a diagnostic.
print(True + 1, 1 + True, True + True, False + 1)
print(True - False, True * 3, 3 * True, True * True)
print(True / 2, 2 / True, True // 1, True % 2)
print(True**2, 2**True)

# Mixed with the rung above: bool -> int -> float.
print(True + 1.5, 1.5 + True, True / 2.0)

# Shifts always answer int, in CPython too.
print(True << 1, True >> 1, 1 << True)

# ⭐ The three that stay BOOL when both operands are bools. `True | 1` is 1,
# `True | False` is True -- same operator, different answer type.
print(True | False, True & True, True ^ True)
print(True | 1, True & 1, True ^ 1, 1 | True)

# Comparisons, which reach int's operators the same way.
print(True < 2, True == 1, True > False, True != 0, True <= 1.0, 1 < True)
# Two bools compare through their truth bits and never touch int.
print(True == True, True != False, False < True)

# Unary. `not` is the one that answers a bool.
print(-True, +True, -False, not True, not False)


def count_hits(flags: list[bool]) -> int:
    total = 0
    for f in flags:
        total += f
    return total


print(count_hits([True, False, True, True]))
print(sum([True, False, True]))
