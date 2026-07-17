"""** is real exponentiation (it used to fall through to __add__), big
integer literals are accepted, and int(str)/int(float)/int(int) convert
with CPython semantics."""

print(2 ** 3)
print(2 ** 100)
print((-3) ** 3)
print((-3) ** 4)
print(0 ** 0)
print(1 ** 1000000000000000000000)
print((-1) ** 1000000000000000000001)
print(10 ** 50)
print(123456789012345678901234567890)
print(-123456789012345678901234567890)
print(123456789012345678901234567890 + 1)
print(int("12345"))
print(int("-987654321098765432109876543210"))
print(int("  +42  "))
print(int("1_000_000"))
print(int(5))
print(int(2.9))
print(int(-2.9))
print(int("0"))
