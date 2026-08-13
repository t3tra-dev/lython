# `int(s, base)`. Only the one-argument form parsed, so `int("ff", 16)` was
# "builtins.int does not provide manifest method '__init__'" -- pointing at
# construction for a call CPython answers by parsing. The parse is a
# synthesized Python function over str and int surface that already compiles,
# so the arbitrary-precision accumulate comes for free (the 16-digit hex
# below overflows i64).
print(int("ff", 16))
print(int("0x1f", 16))
print(int("-101", 2))
print(int("777", 8))
print(int("z", 36))
print(int("  42  ", 10))
print(int("1_000", 10))
print(int("+7", 10))
print(int("0x_1f", 16))
print(int("7fffffffffffffffff", 16))
print(int("ZZ", 36))
print(int("0B1010", 2))

base = 8
print(int("17", base))
print(int("10", base + 8))
print(int("ff", 16) + 1)


# The refusals, message for message with CPython.
def bad(s: str, b: int) -> None:
    try:
        print(int(s, b))
    except ValueError as e:
        print("ValueError:", str(e))


bad("  zz  ", 16)
bad("+", 10)
bad("0x", 16)
bad("_", 10)
bad("1__0", 10)
bad("1_", 10)
bad("-", 2)
bad("10", 1)
bad("10", 37)
bad("", 10)
bad("8", 8)


# The one-argument form is untouched.
print(int("42"))
print(int("-7"))
