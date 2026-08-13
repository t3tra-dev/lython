# `float("2.5")`. Only float(int) and float(float) parsed, so the one spelling
# that reads a number out of text was "builtins.float does not provide
# manifest method '__init__'" -- construction, for a call CPython answers by
# parsing, while `int("42")` right next to it worked.
#
# The digits go to strtod, which is correctly rounded, rather than to an int
# mantissa divided by a power of ten, which rounds twice.
print(float("2.5"))
print(float("-3.25"))
print(float("1e10"))
print(float("  0.5  "))
print(float("1_000.5"))
print(float("42"))
print(float("+.5"))
print(float(".5e2"))
print(float("inf"))
print(float("-Infinity"))
print(float("2.5") + 1.0)
print(float("0.1") + float("0.2"))

text = "6.25"
print(float(text) * 2.0)

# float(int) and float(float) are untouched.
print(float(7))
print(float(2.5))


# What strtod accepts and float() does not, plus the underscore rule.
def bad(s: str) -> None:
    try:
        print(float(s))
    except ValueError as e:
        print("ValueError:", str(e))


bad("0x1p3")
bad("2.5x")
bad("")
bad("   ")
bad("_1")
bad("1__0")
bad("1_")
bad("1.5.2")
bad("e5")
