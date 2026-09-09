# An unannotated parameter takes its type from the call sites, and the walk
# that reads a function body typed the expressions it needed an ANSWER from --
# an assigned value, a return, a yield -- and left the ones it only needed the
# EFFECT of untyped. A call in a CONDITION is one of those, and its effect is
# what binds the callee's parameter:
#
#     def is_leap(y):
#         return y % 4 == 0
#     def month_len(y, m):
#         if is_leap(y):
#             return 29
#         return 30
#     print(month_len(2024, 2))
#     # function parameter 'y' requires an annotation
#
# The SAME call one position over -- `return is_leap(y)`, `flag = is_leap(y)`,
# `int(is_leap(y))` -- resolved it. `while`, `assert`, a ternary's condition
# and a bare expression statement were the same hole.
#
# Why execution: the answers are the point. `is_leap` decides how long February
# is, `overdue` decides which rows are kept, and a parameter typed wrong or a
# body refused would show up as either.
#
# ⭐ The test is typed for the EFFECT and the answer thrown away: a condition's
# type is not the statement's, and recording it would put a bool where the walk
# collects yields and returns.
#
# ⛔ A dead branch is still a call site -- `if False: helper(1)` types the
# parameter -- because the walk reads call sites syntactically. Nothing here
# depends on that, but it is the same rule seen from the other side.


def is_leap(y):
    if y % 400 == 0:
        return True
    if y % 100 == 0:
        return False
    return y % 4 == 0


def month_len(y, m):
    lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if m == 2 and is_leap(y):
        return 29
    return lengths[m - 1]


def days_before(y, m):
    n = 0
    k = 1
    while k < m:
        n += month_len(y, k)
        k += 1
    return n


def overdue(day, limit):
    return day > limit


def keep(days, limit):
    out = []
    for d in days:
        if overdue(d, limit):
            out.append(d)
    return out


def label(day, limit):
    return "late" if overdue(day, limit) else "ok"


def touch(day):
    assert overdue(day, 0)
    return day


print(is_leap(2000), is_leap(1900), is_leap(2024))
print(month_len(2024, 2), month_len(2023, 2), month_len(2024, 1))
print(days_before(2024, 3), days_before(2023, 3))
print(keep([10, 40, 90], 30))
print(label(10, 30), label(90, 30))
print(touch(5))
