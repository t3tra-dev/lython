# What this pins: the `assert` statement.
#
#     assert n > 0, "must be positive"
#     # emit error: unsupported statement kind 'Assert'
#
# It is written as what it means -- `if not test: raise AssertionError(msg)` --
# so nothing new reaches the dialect. Both halves already worked: `raise
# AssertionError` instantiates a raised class with no arguments (which is its own
# fix, recorded on the Raise arm) and `raise AssertionError(m)` is an ordinary
# raise. `builtins.AssertionError` was already in the runtime taxonomy.
#
# Why this needs to run rather than assert on a diagnostic: the point of the
# statement is the side effect of a FAILING test, and the shape it lowers to has
# two arms. A rewrite that inverted the condition would compile and pass every
# true assertion while never raising -- so each section below has a passing and a
# failing assertion over the same expression.
#
# ⛔ Not elided under any flag. CPython drops asserts under -O, Lython has no -O,
# and the CPython default is that they run; an assert that silently did not check
# would be the opposite of this project's rule.
#
# ⛔ The failures are CAUGHT here rather than left to propagate, because an
# uncaught traceback differs from CPython's in a way that has nothing to do with
# assert: CPython underlines the failing expression with `^^^^` markers and
# Lython does not print them.
#
# Every expected line is python3.14's.

# --- the passing forms, with and without a message -------------------------
assert 1 + 1 == 2
assert True, "never"
assert [1], "never"
assert "a" in "abc"
print("passed")


# --- the failing forms, both spellings ------------------------------------
try:
    assert 1 + 1 == 3
except AssertionError as e:
    print("bare:", repr(str(e)))

try:
    assert 1 + 1 == 3, "arithmetic broke"
except AssertionError as e:
    print("with message:", e)


# --- inside a function, where the failure leaves the frame ----------------
def check(n: int) -> int:
    assert n > 0, "must be positive"
    return n * 2


print(check(3), check(1))
try:
    print(check(-1))
except AssertionError as e:
    print("caught:", e)


# --- a computed message ---------------------------------------------------
n = 5
try:
    assert n % 2 == 0, "n=" + str(n)
except AssertionError as e:
    print("computed:", e)


# --- in a loop, and after it ---------------------------------------------
total = 0
for v in [1, 2, 3]:
    assert v > 0
    total += v
assert total == 6, "sum"
print("total", total)

failures = 0
for v in [1, -2, 3]:
    try:
        assert v > 0, "bad " + str(v)
    except AssertionError as e:
        failures += 1
        print("loop:", e)
print("failures", failures)


# --- AssertionError is an Exception, so a broad handler sees it -----------
try:
    assert False, "broad"
except Exception as e:
    print("as Exception:", e)
